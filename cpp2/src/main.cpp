/*
  stream_tflite.cpp
  Realtime PortAudio streamer using TensorFlow Lite (LiteRT) model windows.

  Updated for streaming/stateful realtime TFLite (T=32):
    - Detects realtime_state_* inputs at runtime.
    - If streaming: feeds/updates states via a packed bundle + splits outputs, and
      schedules true streaming blocks (first 'hop' samples filled, rest zeros).
    - If not streaming: falls back to original windowed schedule (left=T-hop).

  Build notes (example):
    clang++ -O3 -std=c++17 stream_tflite.cpp -o streamer_tflite \
      -ltensorflowlite -lXNNPACK -lportaudio -framework Accelerate -lpthread

  Usage example:
    ./streamer_tflite --tflite out/models/realtime/piezo2mic_baseline_realtime_t32_fp32.tflite \
                      --input-device <id> --output-device <id> --blocksize 16
*/

#include <iostream>
#include <vector>
#include <array>
#include <thread>
#include <atomic>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <csignal>
#include <cstring>
#include <cassert>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>
#include <cmath>
#include <sys/stat.h>
#include <unordered_map>
#include <string>
#include <algorithm>

#include <pthread/qos.h>
#include <Accelerate/Accelerate.h>
#include <portaudio.h>
#ifdef __APPLE__
  #include <sched.h>
  #include "pa_mac_core.h"
  #include <CoreAudio/CoreAudio.h>
  #include <CoreFoundation/CoreFoundation.h>
#endif

// TensorFlow Lite
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#ifdef TFLITE_WITH_XNNPACK
  #include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#endif

// ───────────────────────── GLOBALS/FLAGS ─────────────────────────
// Rolling window of style prob (weighted by coreLen)
const double windowSec = 2.0;
const double maxKnobDbChange = 32.0f;

std::atomic<bool> gStopFlag(false), gBypass(false);
std::atomic<uint64_t> gBypassEpoch{0};
void signalHandler(int){ gStopFlag = true; }

enum class StyleMode : int { Auto = 0, ForcePick = 1, ForceFinger = 2 };
std::atomic<int> gStyleMode{ static_cast<int>(StyleMode::Auto) };
static inline const char* styleModeName(int m) {
  switch (static_cast<StyleMode>(m)) {
    case StyleMode::Auto: return "AUTO";
    case StyleMode::ForcePick: return "FORCE_PICK";
    case StyleMode::ForceFinger: return "FORCE_FINGER";
    default: return "?";
  }
}

std::atomic<bool> gGuidedAuto(false);

// Round up to next power of two
static inline size_t nextPow2(size_t x) { size_t p=1; while (p<x) p<<=1; return p; }

// Rolling weighted average over a circular buffer of fixed power-of-two capacity
struct ProbWindow {
  std::vector<float> pbuf;
  std::vector<float> wbuf;
  size_t mask = 0;
  size_t pos  = 0;
  double sumP = 0.0;   // sum of p*w
  double sumW = 0.0;   // sum of w

  void init(size_t capPow2) {
    size_t cap = capPow2 ? capPow2 : 1;
    pbuf.assign(cap, 0.5f);
    wbuf.assign(cap, 0.0f);
    mask = cap - 1;
    pos  = 0;
    sumP = 0.0;
    sumW = 0.0;
  }
  void reset() {
    std::fill(pbuf.begin(), pbuf.end(), 0.5f);
    std::fill(wbuf.begin(), wbuf.end(), 0.0f);
    pos  = 0;
    sumP = 0.0;
    sumW = 0.0;
  }
  // push one sample with weight (e.g., coreLen)
  void push(float p, float w) {
    const size_t i = pos & mask;
    // remove old
    sumP -= double(pbuf[i]) * double(wbuf[i]);
    sumW -= double(wbuf[i]);
    // insert new
    pbuf[i] = p;
    wbuf[i] = std::max(0.0f, w);
    sumP += double(pbuf[i]) * double(wbuf[i]);
    sumW += double(wbuf[i]);
    ++pos;
  }
  float mean() const {
    return (sumW > 1e-9 ? float(sumP / sumW) : 0.5f);
  }
};

// UI knobs (atomic so audio thread reads lock-free)
struct Knobs {
  std::atomic<float> pickAttack_dB{0.0f};
  std::atomic<float> pickReso_dB{0.0f};
  std::atomic<float> ir_dB{0.0f};
  std::atomic<float> wet_pct{100.0f}; // 0..100

  // Output gain (dB → converted to linear for TFLite input)
  std::atomic<float> outGain_dB{0.0f};
  // Phase invert switch: 0.0 = normal, 1.0 = inverted
  std::atomic<float> phaseInvert01{0.0f};

  // Low-cut (high-pass)
  std::atomic<float> lc_hz{20.0f};
  std::atomic<float> lc_slope_dbpoct{6.0f};

  // Low shelf
  std::atomic<float> ls_gain_db{0.0f};
  std::atomic<float> ls_hz{200.0f};
  std::atomic<float> ls_slope_dbpoct{12.0f};

  // Mid bell
  std::atomic<float> mid_gain_db{0.0f};
  std::atomic<float> mid_hz{1000.0f};
  std::atomic<float> mid_q{1.0f};

  // High shelf
  std::atomic<float> hs_gain_db{0.0f};
  std::atomic<float> hs_hz{4000.0f};
  std::atomic<float> hs_slope_dbpoct{12.0f};
} gKnobs;

// For latency logging
static double gIOInputLatencySec  = 0.0;
static double gIOOutputLatencySec = 0.0;
static int    PROCESSING_LOGGING_WAIT_SEC = 2;



// ───────────────────────── UTIL ─────────────────────────
static inline float db2lin(float dB) {
  return std::pow(10.0f, dB / 20.0f);
}
static inline float clampf(float x, float lo, float hi) {
  return std::max(lo, std::min(hi, x));
}







// ───────────────────────── KNOB WEB UI (SEPARATE WINDOW VIA BROWSER) ─────────────────────────
// Add these functions anywhere after the globals (they reference gKnobs, gStyleMode, gBypass, gGuidedAuto).
// Call StartKnobWebUI(8088); from main() after printing the "Keys" help (e.g., right before opening PA stream).
// Optionally call StopKnobWebUI() during shutdown (it also stops automatically when gStopFlag is set).

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>

static std::atomic<bool> gWebUIRunning{false};
static std::atomic<bool> gWebUIShouldStop{false};
static int gWebUIListenFd = -1;
static std::thread gWebUIThread;

static inline void ui_writeAll(int fd, const char* data, size_t len) {
  size_t off = 0;
  while (off < len) {
    ssize_t n = ::send(fd, data + off, len - off, 0);
    if (n <= 0) return;
    off += size_t(n);
  }
}
static inline void ui_writeAllStr(int fd, const std::string& s) {
  ui_writeAll(fd, s.data(), s.size());
}

static std::string ui_styleModeName_jsonsafe() {
  int m = gStyleMode.load();
  const char* n = styleModeName(m);
  std::string s = "\"";
  for (const char* p = n; *p; ++p) {
    char c = *p;
    if (c=='"' || c=='\\') { s.push_back('\\'); s.push_back(c); }
    else s.push_back(c);
  }
  s.push_back('"');
  return s;
}

static std::string ui_htmlIndex() {
  static const char* kHtml = R"HTML(<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Streamer Knobs</title>
<style>
  :root {
    --bg: #16181d; --panel: #21242c; --accent: #ffcc00; --accent2: #00d1ff;
    --text: #eaeef3; --muted: #9aa3ad; --btn: #2a2f3a; --btnOn: #355c3b; --btnOff: #533c3c;
  }
  html, body { margin:0; padding:0; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; user-select:none; }
  .wrap { display:flex; flex-direction:column; align-items:center; gap:14px; padding:14px; }
  .row { display:flex; gap:16px; flex-wrap:wrap; align-items:center; justify-content:center; }
  .pedal {
    background: linear-gradient(180deg, #2a2f3a, #1b1f26); border-radius: 16px;
    box-shadow: 0 10px 24px rgba(0,0,0,0.35), inset 0 2px 0 rgba(255,255,255,0.05);
    padding: 14px 16px; display:flex; gap:18px; align-items:center; justify-content:center; flex-wrap: wrap;
  }
  .knob { width: 130px; height: 170px; display:flex; flex-direction:column; align-items:center; justify-content:flex-start;
    background: rgba(255,255,255,0.02); border-radius: 12px; padding: 8px 10px; }
  .knob canvas { width:110px; height:110px; image-rendering: crisp-edges; }
  .knob .label { font-size: 13px; color: var(--muted); margin-top: 6px; letter-spacing: 0.5px; }
  .knob .value { font-size: 16px; margin-top: 4px; }
  .btn { background: var(--btn); color: var(--text); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px; padding: 10px 14px; cursor: pointer; transition: transform 0.02s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.05); }
  .btn:active { transform: translateY(1px); }
  .btn.on { background: #2f4634; border-color: rgba(0,255,128,0.2); }
  .btn.off{ background: #3f2a2a; border-color: rgba(255,64,64,0.2); }
  .seg { display:flex; gap:8px; background: #14171d; border-radius: 10px; padding: 6px; border: 1px solid rgba(255,255,255,0.06); }
  .seg .segbtn { padding: 8px 12px; cursor: pointer; border-radius: 8px; color: var(--muted); border: 1px solid transparent; background: transparent; }
  .seg .segbtn.active { background: #283145; border-color: rgba(0,128,255,0.4); color: var(--text); box-shadow: inset 0 1px 0 rgba(255,255,255,0.05); }
  .title { font-size: 18px; font-weight: 600; color: var(--muted); letter-spacing: 0.6px; }
  .small { font-size: 12px; color: var(--muted); }
  .grid { display:grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(180px,1fr)); width: 100%; max-width: 1100px; }
  .panel { background: var(--panel); border-radius: 12px; padding: 12px; border: 1px solid rgba(255,255,255,0.06);
           box-shadow: 0 8px 18px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.03); }
  .status { display:flex; gap: 14px; align-items:center; flex-wrap: wrap; }
  .status .kv { color: var(--muted); }
  .hrow { display:flex; gap:16px; flex-wrap:wrap; }
</style>
</head>
<body>
<div class="wrap">
  <div class="row"><div class="title">Realtime Streamer Controls</div></div>

  <!-- Stems section -->
  <div class="row pedal">
    <div class="knob">
      <canvas id="knob_pa" width="110" height="110"></canvas>
      <div class="label">Pick Attack</div>
      <div class="value"><span id="val_pa">0</span> dB</div>
    </div>
    <div class="knob">
      <canvas id="knob_pr" width="110" height="110"></canvas>
      <div class="label">Pick Resonance</div>
      <div class="value"><span id="val_pr">0</span> dB</div>
    </div>
    <div class="knob">
      <canvas id="knob_ir" width="110" height="110"></canvas>
      <div class="label">IR Gain</div>
      <div class="value"><span id="val_ir">0</span> dB</div>
    </div>
    <div class="knob">
      <canvas id="knob_wet" width="110" height="110"></canvas>
      <div class="label">Wet</div>
      <div class="value"><span id="val_wet">100</span> %</div>
    </div>
  </div>

  <!-- Output/Phase -->
  <div class="grid">
    <div class="panel">
      <div class="row"><div class="title">Output / Phase</div></div>
      <div class="hrow">
        <div class="knob">
          <canvas id="knob_outg" width="110" height="110"></canvas>
          <div class="label">Output Gain</div>
          <div class="value"><span id="val_outg">0</span> dB</div>
        </div>
        <div class="panel" style="display:flex;align-items:center;justify-content:center;gap:10px;min-width:200px;">
          <div class="btn" id="btnPhase">Phase Invert: OFF</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Master EQ -->
  <div class="grid">
    <div class="panel">
      <div class="row"><div class="title">Master EQ</div></div>
      <div class="hrow">
        <div class="knob">
          <canvas id="knob_lc_hz" width="110" height="110"></canvas>
          <div class="label">Low Cut Hz</div>
          <div class="value"><span id="val_lc_hz">20</span> Hz</div>
        </div>
        <div class="knob">
          <canvas id="knob_lc_slope" width="110" height="110"></canvas>
          <div class="label">Low Cut Slope</div>
          <div class="value"><span id="val_lc_slope">6</span> dB/oct</div>
        </div>
      </div>

      <div class="hrow">
        <div class="knob">
          <canvas id="knob_ls_gain" width="110" height="110"></canvas>
          <div class="label">Low Shelf Gain</div>
          <div class="value"><span id="val_ls_gain">0</span> dB</div>
        </div>
        <div class="knob">
          <canvas id="knob_ls_hz" width="110" height="110"></canvas>
          <div class="label">Low Shelf Hz</div>
          <div class="value"><span id="val_ls_hz">200</span> Hz</div>
        </div>
        <div class="knob">
          <canvas id="knob_ls_slope" width="110" height="110"></canvas>
          <div class="label">Low Shelf Slope</div>
          <div class="value"><span id="val_ls_slope">12</span> dB/oct</div>
        </div>
      </div>

      <div class="hrow">
        <div class="knob">
          <canvas id="knob_mid_gain" width="110" height="110"></canvas>
          <div class="label">Mid Gain</div>
          <div class="value"><span id="val_mid_gain">0</span> dB</div>
        </div>
        <div class="knob">
          <canvas id="knob_mid_hz" width="110" height="110"></canvas>
          <div class="label">Mid Hz</div>
          <div class="value"><span id="val_mid_hz">1000</span> Hz</div>
        </div>
        <div class="knob">
          <canvas id="knob_mid_q" width="110" height="110"></canvas>
          <div class="label">Mid Q</div>
          <div class="value"><span id="val_mid_q">1.00</span></div>
        </div>
      </div>

      <div class="hrow">
        <div class="knob">
          <canvas id="knob_hs_gain" width="110" height="110"></canvas>
          <div class="label">High Shelf Gain</div>
          <div class="value"><span id="val_hs_gain">0</span> dB</div>
        </div>
        <div class="knob">
          <canvas id="knob_hs_hz" width="110" height="110"></canvas>
          <div class="label">High Shelf Hz</div>
          <div class="value"><span id="val_hs_hz">4000</span> Hz</div>
        </div>
        <div class="knob">
          <canvas id="knob_hs_slope" width="110" height="110"></canvas>
          <div class="label">High Shelf Slope</div>
          <div class="value"><span id="val_hs_slope">12</span> dB/oct</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Mode/Bypass -->
  <div class="grid">
    <div class="panel">
      <div class="row">
        <div class="seg" id="modeSeg">
          <div class="segbtn" data-mode="0">AUTO</div>
          <div class="segbtn" data-mode="1">FORCE_PICK</div>
          <div class="segbtn" data-mode="2">FORCE_FINGER</div>
        </div>
        <div class="btn" id="btnGuided">Guided AUTO: OFF</div>
        <div class="btn" id="btnBypass">Bypass: OFF</div>
      </div>
      <div class="row status" style="margin-top:10px;">
        <div class="kv">Mode: <span id="modeName">AUTO</span></div>
        <div class="kv">Guided: <span id="guidedState">OFF</span></div>
        <div class="kv">Bypass: <span id="bypassState">OFF</span></div>
        <div class="small">(Console prints show current playing style and telemetry.)</div>
      </div>
    </div>
  </div>
</div>

<script>
const DB_MAX = __MAX_DB__;
function clamp(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }
function lerp(a,b,t){ return a + (b-a)*t; }

class Knob {
  constructor(canvas, opts){
    this.c = canvas;
    this.ctx = canvas.getContext('2d');
    this.min = opts.min;
    this.max = opts.max;
    this.step = opts.step || 1;
    this.value = opts.value || 0;
    this.onchange = opts.onchange || function(){};
    this.oninput  = opts.oninput || function(){};
    this.displayEl= opts.displayEl;
    this.prec = opts.prec || 0;
    this.dragging = false;
    this.lastY = 0;
    this.sens = opts.sens || 0.5;
    this.color = opts.color || '#ffcc00';
    this.draw();
    this.c.addEventListener('mousedown', e => this.startDrag(e));
    window.addEventListener('mousemove', e => this.moveDrag(e));
    window.addEventListener('mouseup', e => this.endDrag(e));
    this.c.addEventListener('wheel', e => this.onWheel(e), {passive:false});
    this.c.addEventListener('dblclick', e => this.onDoubleClick(e));
    window.addEventListener('resize', () => this.draw());
  }
  set(v, fire=true){
    const nv = Math.round(clamp(v, this.min, this.max)/this.step)*this.step;
    if (nv !== this.value){
      this.value = nv;
      if (this.displayEl) this.displayEl.textContent = nv.toFixed(this.prec);
      this.draw();
      if (fire) this.onchange(this.value);
    } else {
      this.draw();
      if (fire) this.oninput(this.value);
    }
  }
  startDrag(e){
    e.preventDefault();
    this.dragging = true;
    this.lastY = e.clientY;
    this.lastX = e.clientX;
    this.startVal = this.value;
    this._pxAccum = 0;
  }

  moveDrag(e){
    if (!this.dragging) return;
    const dx = e.clientX - this.lastX;
    const dy = this.lastY - e.clientY; // up = positive (increase), down = negative
    this.lastX = e.clientX;
    this.lastY = e.clientY;

    // Use dominant axis (horizontal or vertical) so dragging left/right also works.
    const useX = Math.abs(dx) > Math.abs(dy);
    const pix = useX ? dx : dy;

    // Make coarse-stepped knobs (like slopes) more responsive by scaling sensitivity
    const range = this.max - this.min;
    const stepNorm = (this.step && range > 0) ? (this.step / range) : 0;
    let effSens = this.sens;
    if (stepNorm >= 0.1) {
      // If step is a large fraction of range (e.g. 6 over 30), boost sensitivity
      effSens = this.sens / stepNorm;
    }

    const delta = pix * effSens * range / 200.0;
    this.set(this.value + delta, true);
  }
  endDrag(e){ if (!this.dragging) return; this.dragging = false; }
  onWheel(e){ e.preventDefault(); const delta = (e.deltaY < 0 ? this.step : -this.step); this.set(this.value + delta, true); }
  onDoubleClick(e){
    // reset to sensible default
    const id = this.c.id;
    const defaults = {
      'knob_pa': 0, 'knob_pr': 0, 'knob_ir': 0, 'knob_wet': 100,
      'knob_outg': 0,
      'knob_lc_hz': 20, 'knob_lc_slope': 6,
      'knob_ls_gain': 0, 'knob_ls_hz': 200, 'knob_ls_slope': 12,
      'knob_mid_gain': 0, 'knob_mid_hz': 1000, 'knob_mid_q': 1.0,
      'knob_hs_gain': 0, 'knob_hs_hz': 4000, 'knob_hs_slope': 12
    };
    const defv = (id in defaults) ? defaults[id] : ( (this.min === 0 && this.max === 100) ? 100 : 0 );
    this.set(defv, true);
  }
  draw(){
    const ctx = this.ctx, w=this.c.width, h=this.c.height;
    ctx.clearRect(0,0,w,h);
    const cx = w/2, cy = h/2, R = Math.min(w,h)/2 - 6;

    const g = ctx.createLinearGradient(0,0,0,h);
    g.addColorStop(0, '#3b4454'); g.addColorStop(1, '#1e232c');
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI*2); ctx.fill();

    ctx.strokeStyle = '#13161b'; ctx.lineWidth = 12; ctx.beginPath(); ctx.arc(cx, cy, R-8, Math.PI*0.7, Math.PI*2.3); ctx.stroke();

    const t = (this.value - this.min) / (this.max - this.min);
    const ang = lerp(Math.PI*0.7, Math.PI*2.3, t);

    ctx.strokeStyle = this.color; ctx.lineWidth = 12; ctx.lineCap='round';
    ctx.beginPath(); ctx.arc(cx, cy, R-8, Math.PI*0.7, ang); ctx.stroke();

    ctx.save(); ctx.translate(cx, cy); ctx.rotate(Math.PI*0.7);
    const ticks = 12;
    for (let i=0;i<=ticks;i++){
      const a = (Math.PI*1.6) * (i/ticks);
      ctx.save(); ctx.rotate(a);
      ctx.fillStyle = i/ticks <= t ? this.color : '#303742';
      ctx.fillRect(R-18, -1, 10, 2);
      ctx.restore();
    }
    ctx.restore();

    ctx.save(); ctx.translate(cx,cy); ctx.rotate(ang);
    ctx.fillStyle = '#0c0f13'; ctx.fillRect(R-22, -3, 14, 6); ctx.restore();

    const g2 = ctx.createRadialGradient(cx,cy,2, cx,cy, R/2);
    g2.addColorStop(0, '#232a33'); g2.addColorStop(1, '#0f1318');
    ctx.fillStyle = g2; ctx.beginPath(); ctx.arc(cx, cy, R/2.2, 0, Math.PI*2); ctx.fill();

    ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(cx,cy,R-5, -Math.PI*0.35, Math.PI*0.35); ctx.stroke();
  }
}

const el = (id)=>document.getElementById(id);
const $modeSeg = el('modeSeg');
const $btnGuided = el('btnGuided');
const $btnBypass = el('btnBypass');
const $btnPhase  = el('btnPhase');

const kb = {
  pa: new Knob(el('knob_pa'),  {min:-DB_MAX,max:DB_MAX,step:1,value:0,prec:0,displayEl:el('val_pa'),color:'#ffcc00', onchange:(v)=>sendSet({pa:v})}),
  pr: new Knob(el('knob_pr'),  {min:-DB_MAX,max:DB_MAX,step:1,value:0,prec:0,displayEl:el('val_pr'),color:'#ff7f50', onchange:(v)=>sendSet({pr:v})}),
  ir: new Knob(el('knob_ir'),  {min:-DB_MAX,max:DB_MAX,step:1,value:0,prec:0,displayEl:el('val_ir'),color:'#7ee787', onchange:(v)=>sendSet({ir:v})}),
  wet:new Knob(el('knob_wet'), {min:0,max:100,step:1,value:100,prec:0,displayEl:el('val_wet'),color:'#00d1ff', onchange:(v)=>sendSet({wet:v})}),

  outg: new Knob(el('knob_outg'), {min:-24,max:24,step:1,value:0,prec:0,displayEl:el('val_outg'),color:'#c586f7', onchange:(v)=>sendSet({outg:v})}),

  lc_hz:    new Knob(el('knob_lc_hz'),    {min:10,max:300,step:1,value:20,prec:0,displayEl:el('val_lc_hz'),color:'#e5c07b', onchange:(v)=>sendSet({lc_hz:v})}),
  lc_slope: new Knob(el('knob_lc_slope'), {min:6,max:36,step:6,value:6,prec:0,displayEl:el('val_lc_slope'),color:'#e5c07b', onchange:(v)=>sendSet({lc_slope:v})}),

  ls_gain:  new Knob(el('knob_ls_gain'),  {min:-18,max:18,step:1,value:0,prec:0,displayEl:el('val_ls_gain'),color:'#98c379', onchange:(v)=>sendSet({ls_gain:v})}),
  ls_hz:    new Knob(el('knob_ls_hz'),    {min:50,max:1000,step:5,value:200,prec:0,displayEl:el('val_ls_hz'),color:'#98c379', onchange:(v)=>sendSet({ls_hz:v})}),
  ls_slope: new Knob(el('knob_ls_slope'), {min:6,max:36,step:6,value:12,prec:0,displayEl:el('val_ls_slope'),color:'#98c379', onchange:(v)=>sendSet({ls_slope:v})}),

  mid_gain: new Knob(el('knob_mid_gain'), {min:-18,max:18,step:1,value:0,prec:0,displayEl:el('val_mid_gain'),color:'#61afef', onchange:(v)=>sendSet({mid_gain:v})}),
  mid_hz:   new Knob(el('knob_mid_hz'),   {min:200,max:6000,step:10,value:1000,prec:0,displayEl:el('val_mid_hz'),color:'#61afef', onchange:(v)=>sendSet({mid_hz:v})}),
  mid_q:    new Knob(el('knob_mid_q'),    {min:0.5,max:2.0,step:0.05,value:1.0,prec:2,displayEl:el('val_mid_q'),color:'#61afef', onchange:(v)=>sendSet({mid_q:v})}),

  hs_gain:  new Knob(el('knob_hs_gain'),  {min:-18,max:18,step:1,value:0,prec:0,displayEl:el('val_hs_gain'),color:'#d19a66', onchange:(v)=>sendSet({hs_gain:v})}),
  hs_hz:    new Knob(el('knob_hs_hz'),    {min:2000,max:16000,step:50,value:4000,prec:0,displayEl:el('val_hs_hz'),color:'#d19a66', onchange:(v)=>sendSet({hs_hz:v})}),
  hs_slope: new Knob(el('knob_hs_slope'), {min:6,max:36,step:6,value:12,prec:0,displayEl:el('val_hs_slope'),color:'#d19a66', onchange:(v)=>sendSet({hs_slope:v})}),
};

function sendSet(obj) {
  const params = new URLSearchParams();
  if ('pa' in obj) params.set('pa', obj.pa);
  if ('pr' in obj) params.set('pr', obj.pr);
  if ('ir' in obj) params.set('ir', obj.ir);
  if ('wet' in obj) params.set('wet', obj.wet);

  if ('outg' in obj) params.set('outg', obj.outg);

  if ('lc_hz' in obj) params.set('lc_hz', obj.lc_hz);
  if ('lc_slope' in obj) params.set('lc_slope', obj.lc_slope);

  if ('ls_gain' in obj) params.set('ls_gain', obj.ls_gain);
  if ('ls_hz' in obj) params.set('ls_hz', obj.ls_hz);
  if ('ls_slope' in obj) params.set('ls_slope', obj.ls_slope);

  if ('mid_gain' in obj) params.set('mid_gain', obj.mid_gain);
  if ('mid_hz' in obj) params.set('mid_hz', obj.mid_hz);
  if ('mid_q' in obj) params.set('mid_q', obj.mid_q);

  if ('hs_gain' in obj) params.set('hs_gain', obj.hs_gain);
  if ('hs_hz' in obj) params.set('hs_hz', obj.hs_hz);
  if ('hs_slope' in obj) params.set('hs_slope', obj.hs_slope);

  if ('phase' in obj) params.set('phase', obj.phase ? 1 : 0);

  fetch('/set?' + params.toString()).catch(()=>{});
}

function setMode(m) { fetch('/mode?v='+m).catch(()=>{}); }
function setGuided(on) { fetch('/guided?v='+(on?1:0)).catch(()=>{}); }
function setBypass(on) { fetch('/bypass?v='+(on?1:0)).catch(()=>{}); }
function setPhase(on) { fetch('/set?phase='+(on?1:0)).catch(()=>{}); }

$modeSeg.querySelectorAll('.segbtn').forEach(btn=>{
  btn.addEventListener('click', ()=>{ const v = parseInt(btn.dataset.mode,10); setMode(v); });
});

$btnGuided.addEventListener('click', ()=>{
  const isOn = $btnGuided.classList.contains('on'); setGuided(!isOn);
});
$btnBypass.addEventListener('click', ()=>{
  const isOn = $btnBypass.classList.contains('on'); setBypass(!isOn);
});
$btnPhase.addEventListener('click', ()=>{
  const isOn = $btnPhase.classList.contains('on'); setPhase(!isOn);
});

function applyStatus(s){
  if (typeof s !== 'object') return;
  kb.pa.set(s.pickAttack_dB, false);
  kb.pr.set(s.pickReso_dB, false);
  kb.ir.set(s.ir_dB, false);
  kb.wet.set(s.wet_pct, false);

  kb.outg.set(s.outGain_dB, false);

  kb.lc_hz.set(s.lc_hz, false);
  kb.lc_slope.set(s.lc_slope_dbpoct, false);

  kb.ls_gain.set(s.ls_gain_db, false);
  kb.ls_hz.set(s.ls_hz, false);
  kb.ls_slope.set(s.ls_slope_dbpoct, false);

  kb.mid_gain.set(s.mid_gain_db, false);
  kb.mid_hz.set(s.mid_hz, false);
  kb.mid_q.set(s.mid_q, false);

  kb.hs_gain.set(s.hs_gain_db, false);
  kb.hs_hz.set(s.hs_hz, false);
  kb.hs_slope.set(s.hs_slope_dbpoct, false);

  // Phase switch
  const phaseOn = (s.phaseInvert01 >= 0.5);
  $btnPhase.textContent = 'Phase Invert: ' + (phaseOn ? 'ON' : 'OFF');
  $btnPhase.classList.toggle('on', phaseOn);
  $btnPhase.classList.toggle('off', !phaseOn);

  // Seg
  $modeSeg.querySelectorAll('.segbtn').forEach(b=>{
    if (parseInt(b.dataset.mode,10) === s.mode) b.classList.add('active');
    else b.classList.remove('active');
  });

  el('modeName').textContent = s.modeName;
  el('guidedState').textContent = s.guidedAuto ? 'ON' : 'OFF';
  el('bypassState').textContent = s.bypass ? 'ON' : 'OFF';

  $btnGuided.textContent = 'Guided AUTO: ' + (s.guidedAuto?'ON':'OFF');
  $btnGuided.classList.toggle('on', !!s.guidedAuto);
  $btnGuided.classList.toggle('off', !s.guidedAuto);

  $btnBypass.textContent = 'Bypass: ' + (s.bypass?'ON':'OFF');
  $btnBypass.classList.toggle('on', !!s.bypass);
  $btnBypass.classList.toggle('off', !s.bypass);
}

async function poll(){
  try{
    const r = await fetch('/status?ts=' + Date.now());
    const s = await r.json();
    applyStatus(s);
  } catch (e) {}
}
setInterval(poll, 150);
poll();
</script>
</body>
</html>)HTML";
  return std::string(kHtml);
}

static std::string ui_jsonStatus() {
  std::ostringstream oss;
  oss.setf(std::ios::fixed); oss<<std::setprecision(3);
  oss<<"{";
  // existing
  oss<<"\"pickAttack_dB\":"<< gKnobs.pickAttack_dB.load() <<",";
  oss<<"\"pickReso_dB\":"<< gKnobs.pickReso_dB.load() <<",";
  oss<<"\"ir_dB\":"      << gKnobs.ir_dB.load()        <<",";
  oss<<"\"wet_pct\":"    << gKnobs.wet_pct.load()      <<",";

  // new
  oss<<"\"outGain_dB\":"     << gKnobs.outGain_dB.load()          <<",";
  oss<<"\"phaseInvert01\":"  << gKnobs.phaseInvert01.load()       <<",";

  oss<<"\"lc_hz\":"          << gKnobs.lc_hz.load()               <<",";
  oss<<"\"lc_slope_dbpoct\":"<< gKnobs.lc_slope_dbpoct.load()     <<",";

  oss<<"\"ls_gain_db\":"     << gKnobs.ls_gain_db.load()          <<",";
  oss<<"\"ls_hz\":"          << gKnobs.ls_hz.load()               <<",";
  oss<<"\"ls_slope_dbpoct\":"<< gKnobs.ls_slope_dbpoct.load()     <<",";

  oss<<"\"mid_gain_db\":"    << gKnobs.mid_gain_db.load()         <<",";
  oss<<"\"mid_hz\":"         << gKnobs.mid_hz.load()              <<",";
  oss<<"\"mid_q\":"          << gKnobs.mid_q.load()               <<",";

  oss<<"\"hs_gain_db\":"     << gKnobs.hs_gain_db.load()          <<",";
  oss<<"\"hs_hz\":"          << gKnobs.hs_hz.load()               <<",";
  oss<<"\"hs_slope_dbpoct\":"<< gKnobs.hs_slope_dbpoct.load()     <<",";

  // switches/mode
  oss<<"\"bypass\":"     << (gBypass.load()?1:0)       <<",";
  oss<<"\"mode\":"       << gStyleMode.load()          <<",";
  oss<<"\"modeName\":"   << ui_styleModeName_jsonsafe() <<",";
  oss<<"\"guidedAuto\":" << (gGuidedAuto.load()?1:0);
  oss<<"}";
  return oss.str();
}

static inline std::string ui_httpHeader(const char* status, const char* mime, size_t contentLen, bool nocache=false) {
  std::ostringstream oss;
  oss<<"HTTP/1.1 "<<status<<"\r\n";
  oss<<"Connection: close\r\n";
  if (mime) oss<<"Content-Type: "<<mime<<"\r\n";
  oss<<"Content-Length: "<<contentLen<<"\r\n";
  if (nocache) {
    oss<<"Cache-Control: no-cache, no-store, must-revalidate\r\n";
    oss<<"Pragma: no-cache\r\n";
    oss<<"Expires: 0\r\n";
  }
  oss<<"\r\n";
  return oss.str();
}

static inline std::string ui_urlDecode(const std::string& s) {
  std::string out; out.reserve(s.size());
  for (size_t i=0;i<s.size();++i){
    char c = s[i];
    if (c=='%') {
      if (i+2 < s.size()) {
        int hi = std::isdigit(s[i+1]) ? s[i+1]-'0' : (std::tolower(s[i+1])-'a'+10);
        int lo = std::isdigit(s[i+2]) ? s[i+2]-'0' : (std::tolower(s[i+2])-'a'+10);
        if (hi>=0 && hi<16 && lo>=0 && lo<16) {
          out.push_back(char(hi*16 + lo));
          i+=2; continue;
        }
      }
      out.push_back('%');
    } else if (c=='+') out.push_back(' ');
    else out.push_back(c);
  }
  return out;
}

static inline std::unordered_map<std::string,std::string> ui_parseQuery(const std::string& q) {
  std::unordered_map<std::string,std::string> m;
  size_t i=0;
  while (i<q.size()){
    size_t amp = q.find('&', i);
    std::string kv = q.substr(i, amp==std::string::npos ? std::string::npos : amp-i);
    size_t eq = kv.find('=');
    if (eq == std::string::npos) {
      m[ui_urlDecode(kv)] = "";
    } else {
      m[ui_urlDecode(kv.substr(0,eq))] = ui_urlDecode(kv.substr(eq+1));
    }
    if (amp==std::string::npos) break;
    i = amp+1;
  }
  return m;
}

static inline void ui_handleSet(const std::unordered_map<std::string,std::string>& q) {
  auto getf = [&](const char* k, float &dst, float lo, float hi){
    auto it = q.find(k);
    if (it != q.end()) {
      try {
        float v = std::stof(it->second);
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        dst = v;
      } catch (...) {}
    }
  };
  // Existing
  float pa = gKnobs.pickAttack_dB.load();
  float pr = gKnobs.pickReso_dB.load();
  float ir = gKnobs.ir_dB.load();
  float wt = gKnobs.wet_pct.load();
  getf("pa", pa, -maxKnobDbChange, maxKnobDbChange);
  getf("pr", pr, -maxKnobDbChange, maxKnobDbChange);
  getf("ir", ir, -maxKnobDbChange, maxKnobDbChange);
  getf("wet", wt, 0.0f, 100.0f);
  gKnobs.pickAttack_dB.store(pa);
  gKnobs.pickReso_dB.store(pr);
  gKnobs.ir_dB.store(ir);
  gKnobs.wet_pct.store(wt);

  // New: output/phase
  float outg = gKnobs.outGain_dB.load();
  getf("outg", outg, -24.0f, 24.0f);
  gKnobs.outGain_dB.store(outg);

  auto itPh = q.find("phase");
  if (itPh != q.end()) {
    try {
      int v = std::stoi(itPh->second);
      gKnobs.phaseInvert01.store((v!=0) ? 1.0f : 0.0f);
    } catch (...) {}
  }

  // Low cut
  float lc_hz = gKnobs.lc_hz.load();
  float lc_sl = gKnobs.lc_slope_dbpoct.load();
  getf("lc_hz", lc_hz, 10.0f, 300.0f);
  getf("lc_slope", lc_sl, 6.0f, 36.0f);
  gKnobs.lc_hz.store(lc_hz);
  gKnobs.lc_slope_dbpoct.store(lc_sl);

  // Low shelf
  float ls_g = gKnobs.ls_gain_db.load();
  float ls_h = gKnobs.ls_hz.load();
  float ls_s = gKnobs.ls_slope_dbpoct.load();
  getf("ls_gain", ls_g, -18.0f, 18.0f);
  getf("ls_hz",   ls_h, 50.0f,  1000.0f);
  getf("ls_slope",ls_s, 6.0f,   36.0f);
  gKnobs.ls_gain_db.store(ls_g);
  gKnobs.ls_hz.store(ls_h);
  gKnobs.ls_slope_dbpoct.store(ls_s);

  // Mid bell
  float md_g = gKnobs.mid_gain_db.load();
  float md_h = gKnobs.mid_hz.load();
  float md_q = gKnobs.mid_q.load();
  getf("mid_gain", md_g, -18.0f, 18.0f);
  getf("mid_hz",   md_h, 200.0f, 6000.0f);
  getf("mid_q",    md_q, 0.5f,   2.0f);
  gKnobs.mid_gain_db.store(md_g);
  gKnobs.mid_hz.store(md_h);
  gKnobs.mid_q.store(md_q);

  // High shelf
  float hs_g = gKnobs.hs_gain_db.load();
  float hs_h = gKnobs.hs_hz.load();
  float hs_s = gKnobs.hs_slope_dbpoct.load();
  getf("hs_gain", hs_g, -18.0f, 18.0f);
  getf("hs_hz",   hs_h, 2000.0f, 16000.0f);
  getf("hs_slope",hs_s, 6.0f, 36.0f);
  gKnobs.hs_gain_db.store(hs_g);
  gKnobs.hs_hz.store(hs_h);
  gKnobs.hs_slope_dbpoct.store(hs_s);
}

static inline void ui_handleMode(const std::unordered_map<std::string,std::string>& q) {
  auto it = q.find("v");
  if (it != q.end()) {
    try {
      int v = std::stoi(it->second);
      if (v < 0) v = 0; if (v > 2) v = 2;
      gStyleMode.store(v);
    } catch (...) {}
  }
}
static inline void ui_handleGuided(const std::unordered_map<std::string,std::string>& q) {
  auto it = q.find("v");
  if (it != q.end()) {
    gGuidedAuto.store(it->second=="1" || it->second=="true" || it->second=="on");
  }
}

static inline void ui_handleBypass(const std::unordered_map<std::string,std::string>& q) {
  auto it = q.find("v");
  if (it != q.end()) {
    bool on = (it->second=="1" || it->second=="true" || it->second=="on");
    bool prev = gBypass.exchange(on, std::memory_order_acq_rel);
    if (on != prev) {
      gBypassEpoch.fetch_add(1, std::memory_order_acq_rel);
    }
  }
}

static inline void ui_replaceAll(std::string &s, const std::string& from, const std::string& to) {
  if (from.empty()) return;
  size_t pos = 0;
  while ((pos = s.find(from, pos)) != std::string::npos) {
    s.replace(pos, from.size(), to);
    pos += to.size();
  }
}

static void ui_handleClient(int cfd) {
  // Read basic HTTP request (up to headers end). Limit to 16KB
  std::string req; req.resize(0);
  char buf[4096];
  int headerEnd = -1;
  for (;;) {
    ssize_t n = ::recv(cfd, buf, sizeof(buf), 0);
    if (n <= 0) break;
    req.append(buf, buf+n);
    // find \r\n\r\n
    size_t pos = req.find("\r\n\r\n");
    if (pos != std::string::npos) { headerEnd = int(pos + 4); break; }
    if (req.size() > 16384) break;
  }
  if (headerEnd < 0) {
    std::string h = ui_httpHeader("400 Bad Request", "text/plain; charset=utf-8", 0);
    ui_writeAllStr(cfd, h);
    return;
  }
  // Parse request line
  size_t lineEnd = req.find("\r\n");
  if (lineEnd == std::string::npos) {
    std::string h = ui_httpHeader("400 Bad Request", "text/plain; charset=utf-8", 0);
    ui_writeAllStr(cfd, h);
    return;
  }
  std::string firstLine = req.substr(0, lineEnd);
  std::istringstream iss(firstLine);
  std::string method, uri, ver;
  iss >> method >> uri >> ver;
  std::string path = uri, query;
  size_t qpos = uri.find('?');
  if (qpos != std::string::npos) { path = uri.substr(0,qpos); query = uri.substr(qpos+1); }

  // Routes
  if (method == "GET" && (path == "/" || path == "/index.html")) {
    std::string body = ui_htmlIndex();
    ui_replaceAll(body, "__MAX_DB__", std::to_string(maxKnobDbChange));
    std::string h = ui_httpHeader("200 OK", "text/html; charset=utf-8", body.size());
    ui_writeAllStr(cfd, h); ui_writeAllStr(cfd, body);
  } else if (method == "GET" && path == "/status") {
    std::string body = ui_jsonStatus();
    std::string h = ui_httpHeader("200 OK", "application/json; charset=utf-8", body.size(), true);
    ui_writeAllStr(cfd, h); ui_writeAllStr(cfd, body);
  } else if (method == "GET" && path == "/set") {
    auto q = ui_parseQuery(query);
    ui_handleSet(q);
    std::string body = "{\"ok\":1}";
    std::string h = ui_httpHeader("200 OK", "application/json; charset=utf-8", body.size(), true);
    ui_writeAllStr(cfd, h); ui_writeAllStr(cfd, body);
  } else if (method == "GET" && path == "/mode") {
    auto q = ui_parseQuery(query);
    ui_handleMode(q);
    std::string body = "{\"ok\":1}";
    std::string h = ui_httpHeader("200 OK", "application/json; charset=utf-8", body.size(), true);
    ui_writeAllStr(cfd, h); ui_writeAllStr(cfd, body);
  } else if (method == "GET" && path == "/guided") {
    auto q = ui_parseQuery(query);
    ui_handleGuided(q);
    std::string body = "{\"ok\":1}";
    std::string h = ui_httpHeader("200 OK", "application/json; charset=utf-8", body.size(), true);
    ui_writeAllStr(cfd, h); ui_writeAllStr(cfd, body);
  } else if (method == "GET" && path == "/bypass") {
    auto q = ui_parseQuery(query);
    ui_handleBypass(q);
    std::string body = "{\"ok\":1}";
    std::string h = ui_httpHeader("200 OK", "application/json; charset=utf-8", body.size(), true);
    ui_writeAllStr(cfd, h); ui_writeAllStr(cfd, body);
  } else {
    std::string body = "Not Found";
    std::string h = ui_httpHeader("404 Not Found", "text/plain; charset=utf-8", body.size());
    ui_writeAllStr(cfd, h); ui_writeAllStr(cfd, body);
  }
}

static void ui_tryOpenBrowser(uint16_t port) {
#if defined(__APPLE__)
  std::string cmd = "open http://127.0.0.1:" + std::to_string(port) + "/";
  std::system(cmd.c_str());
#elif defined(_WIN32)
  std::string url = "http://127.0.0.1:" + std::to_string(port) + "/";
  std::string cmd = "start " + url;
  std::system(cmd.c_str());
#else
  std::string cmd = "xdg-open http://127.0.0.1:" + std::to_string(port) + " >/dev/null 2>&1 &";
  std::system(cmd.c_str());
#endif
}

static void KnobWebUIServerLoop(uint16_t port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) { std::cerr<<"[UI] socket() failed\n"; return; }
  gWebUIListenFd = fd;

  int yes = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK); // 127.0.0.1
  addr.sin_port = htons(port);

  if (::bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
    std::cerr<<"[UI] bind("<<port<<") failed\n";
    ::close(fd); gWebUIListenFd = -1; return;
  }
  if (::listen(fd, 8) < 0) {
    std::cerr<<"[UI] listen() failed\n";
    ::close(fd); gWebUIListenFd = -1; return;
  }

  std::cout<<"[UI] Web UI on http://127.0.0.1:"<<port<<"/  (opens in your browser)\n";
  ui_tryOpenBrowser(port);

  // Accept loop
  while (!gStopFlag.load() && !gWebUIShouldStop.load()) {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(fd, &rfds);
    timeval tv; tv.tv_sec = 0; tv.tv_usec = 200000;
    int rv = ::select(fd+1, &rfds, nullptr, nullptr, &tv);
    if (rv > 0 && FD_ISSET(fd, &rfds)) {
      sockaddr_in cli{}; socklen_t cl = sizeof(cli);
      int cfd = ::accept(fd, (struct sockaddr*)&cli, &cl);
      if (cfd >= 0) {
        ui_handleClient(cfd);
        ::shutdown(cfd, SHUT_RDWR);
        ::close(cfd);
      }
    }
  }
  ::close(fd);
  gWebUIListenFd = -1;
}

static void StartKnobWebUI(uint16_t port) {
  if (gWebUIRunning.load()) return;
  gWebUIShouldStop.store(false);
  gWebUIRunning.store(true);
  gWebUIThread = std::thread([port](){
    // Lower priority for UI thread
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#endif
    KnobWebUIServerLoop(port);
    gWebUIRunning.store(false);
  });
}

static void StopKnobWebUI() {
  gWebUIShouldStop.store(true);
  if (gWebUIListenFd >= 0) {
    // poke the accept()
    int s = ::socket(AF_INET, SOCK_STREAM, 0);
    if (s >= 0) {
      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
      addr.sin_port = htons( (uint16_t) ( (gWebUIListenFd>=0) ? 0 : 0 ) ); // not used
      // Best-effort connect to wake; we don't know bound port in this snippet, so rely on ShouldStop and select timeout
      ::close(s);
    }
  }
  if (gWebUIThread.joinable()) gWebUIThread.join();
}













// ───────────────────────── RING BUFFER ─────────────────────────
struct RingBuffer {
  std::vector<float> buf;
  size_t mask;
  std::atomic<size_t> r{0}, w{0};
  explicit RingBuffer(size_t cap): buf(cap), mask(cap-1){
    assert((cap & mask)==0); // must be power-of-two
  }
  size_t available() const {
    return (w.load(std::memory_order_acquire)
          - r.load(std::memory_order_acquire)) & mask;
  }
  size_t freeSpace() const {
    return (mask+1) - available();
  }
  size_t push(const float *data, size_t n){
    size_t free = freeSpace(), toW = std::min(n, free);
    size_t w0 = w.load(std::memory_order_relaxed);
    for(size_t i=0;i<toW;i++) buf[(w0+i)&mask] = data[i];
    w.store((w0+toW)&mask, std::memory_order_release);
    return toW;
  }
  size_t pop(float *out, size_t n){
    size_t av = available(), toR = std::min(n,av);
    size_t r0 = r.load(std::memory_order_relaxed);
    for(size_t i=0;i<toR;i++) out[i] = buf[(r0+i)&mask];
    r.store((r0+toR)&mask, std::memory_order_release);
    return toR;
  }
};

struct Shared {
  RingBuffer in, out;
  std::vector<float> tempOut;
  explicit Shared(size_t cap): in(cap), out(cap){}
};

// ───────────────────────── DEVICE ENUM ─────────────────────────
struct DeviceInfo {
  int devId;
  std::string name;
  int maxInputChannels;
  int maxOutputChannels;
  double defaultSampleRate;
};

static std::vector<DeviceInfo> getDevices() {
  PaError err = Pa_Initialize();
  if (err != paNoError) {
    std::cerr << "Pa_Initialize failed: " << Pa_GetErrorText(err) << "\n";
    std::exit(1);
  }
  int n = Pa_GetDeviceCount();
  if (n < 0) {
    std::cerr << "Pa_GetDeviceCount failed: " << Pa_GetErrorText(n) << "\n";
    Pa_Terminate(); std::exit(1);
  }
  std::vector<DeviceInfo> devices;
  devices.reserve(n);
  for (int i = 0; i < n; ++i) {
    const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
    if (!info) continue;
    DeviceInfo d;
    d.devId = i;
    d.name = info->name ? info->name : "";
    d.maxInputChannels  = info->maxInputChannels;
    d.maxOutputChannels = info->maxOutputChannels;
    d.defaultSampleRate = info->defaultSampleRate;
    devices.push_back(d);
  }
  Pa_Terminate();
  return devices;
}

static void listDevices(const std::vector<DeviceInfo> &devices) {
  std::cout<<"PortAudio devices:\n";
  for (const auto &d : devices) {
    std::cout<<"  "<<d.devId<<": "<<d.name
             <<" (inputs: "<<d.maxInputChannels
             <<", outputs: "<<d.maxOutputChannels
             <<", sampleRate: "<<d.defaultSampleRate<<")\n";
  }
}

// ────────────────────── NAME/DTYPE HELPERS ─────────────────────
static inline std::string normName(const char* n) {
  std::string s = n ? std::string(n) : std::string();
  if (s.rfind("serving_default_", 0) == 0)
    s = s.substr(std::strlen("serving_default_"));
  auto pos = s.find(':');
  if (pos != std::string::npos) s = s.substr(0, pos);
  // Remove TF scope prefixes like "StatefulPartitionedCall/..."
  auto slash = s.find_last_of('/');
  if (slash != std::string::npos) s = s.substr(slash+1);
  return s;
}
static inline bool isIntType(TfLiteType t) {
  return (t == kTfLiteInt32 || t == kTfLiteInt64 || t == kTfLiteUInt8 || t == kTfLiteInt16);
}

// ───────────────────────── TFLITE WRAPPER ─────────────────────────
class TFLiteRealtime {
public:
  float getStyleEma() const {
    int idx = findStateSlotIndex("realtime_state_style_ema");
    if (idx < 0) return std::numeric_limits<float>::quiet_NaN();
    const StateSlot& s = stateSlots_[idx];
    if (s.buf.empty()) return std::numeric_limits<float>::quiet_NaN();
    // state is [1,1,1] flattened
    return s.buf[0];
  }

private:
  int findStateSlotIndex(const std::string& name) const {
    for (size_t i = 0; i < stateSlots_.size(); ++i) {
      if (stateSlots_[i].name == name) return static_cast<int>(i);
    }
    return -1;
  }
public:
  explicit TFLiteRealtime(const std::string& modelPath) {
    model_ = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
    if (!model_) {
      std::cerr << "[TFLite] Cannot load model: " << modelPath << "\n";
      std::exit(1);
    }
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model_, resolver)(&interp_);
    if (!interp_) {
      std::cerr << "[TFLite] InterpreterBuilder failed\n";
      std::exit(1);
    }
    interp_->SetNumThreads(1);

    #ifdef TFLITE_WITH_XNNPACK
      {
        TfLiteXNNPackDelegateOptions opt = TfLiteXNNPackDelegateOptionsDefault();
        opt.num_threads = 1; // keep real-time friendly
        xnnpack_ = TfLiteXNNPackDelegateCreate(&opt);
        if (xnnpack_) interp_->ModifyGraphWithDelegate(xnnpack_);
      }
    #endif

    if (interp_->AllocateTensors() != kTfLiteOk) {
      std::cerr << "[TFLite] AllocateTensors failed\n";
      std::exit(1);
    }

    // Build input/output maps
    for (int i : interp_->inputs()) {
      auto* t = interp_->tensor(i);
      in_map_[normName(t->name)] = i;
    }
    for (int i : interp_->outputs()) {
      auto* t = interp_->tensor(i);
      out_map_[normName(t->name)] = i;
    }

    // Required inputs
    in_piezo_   = findInput("realtime_piezo_input");
    in_st_true_ = findInput("realtime_style_true_input");
    in_st_force_= findInput("realtime_style_force_input");
    in_knob_pa_ = findInput("realtime_knob_pickattack");
    in_knob_pr_ = findInput("realtime_knob_pickreso");
    in_knob_ir_ = findInput("realtime_knob_ir");
    in_knob_dw_ = findInput("realtime_knob_drywet");

    // Determine T
    auto* tin = (in_piezo_ >= 0) ? interp_->tensor(in_piezo_) : nullptr;
    if (!tin || tin->dims->size < 2) {
      std::cerr<<"[TFLite] Unexpected/no piezo input; want [1,T,1]\n";
      std::exit(1);
    }
    if (tin->dims->size == 3 && (tin->dims->data[0] != 1 || tin->dims->data[2] != 1)) {
      std::cerr<<"[TFLite] Unexpected piezo shape; want [1,T,1]\n";
      std::exit(1);
    }
    T_ = tin->dims->data[1];

    // Discover streaming states by input names
    discoverStateInputs();

    // Pick outputs by shape/dtype
    classifyOutputsByShape();

    // Basic checks for required non-state I/O
    if (in_piezo_ < 0 || out_audio_ < 0 || in_st_true_ < 0 || in_st_force_ < 0 ||
        in_knob_pa_ < 0 || in_knob_pr_ < 0 || in_knob_ir_ < 0 || in_knob_dw_ < 0) {
      std::cerr << "[TFLite] Missing required I/O tensors (check model names)\n";
      std::exit(1);
    }

    std::cout<<"[TFLite] Ready. T="<<T_<<"  streaming="<<(isStreaming()?"true":"false")
             <<"  states="<<stateSlots_.size()<<"\n";
  }

  ~TFLiteRealtime() {
  #ifdef TFLITE_WITH_XNNPACK
      if (xnnpack_) TfLiteXNNPackDelegateDelete(xnnpack_);
  #endif
  }

  int T() const { return T_; }
  bool isStreaming() const { return !stateSlots_.empty(); }

  // Reset state buffers (zeros); call before starting stream
  void resetStates() {
    for (auto &s : stateSlots_) {
      std::fill(s.buf.begin(), s.buf.end(), 0.0f);
    }
  }

  // STREAMING invoke:
  //   - coreLen samples are at the start of 'inBlock' (size >= coreLen),
  //     we fill the first coreLen of [1,T,1] and zero the remainder.
  //   - Returns the full T-length output block; caller will use first coreLen samples.
  void invokeStreamingBlock(const float* inBlockCore, int coreLen,
                          float* outBlockT, float &styleProb,
                          float pa_lin, float pr_lin, float ir_lin, float dry_frac,
                          float style_true, float style_force)
{
  // Prepare piezo [1,T,1]
  float* pz = interp_->typed_tensor<float>(in_piezo_);
  std::fill(pz, pz + T_, 0.0f);
  if (coreLen > 0) std::memcpy(pz, inBlockCore, sizeof(float) * std::min(coreLen, T_));

  // Scalars [1,1] for style (may be guided)
  *interp_->typed_tensor<float>(in_st_true_)  = style_true;
  *interp_->typed_tensor<float>(in_st_force_) = style_force;

  // Stems knobs (linear already)
  *interp_->typed_tensor<float>(in_knob_pa_) = pa_lin;
  *interp_->typed_tensor<float>(in_knob_pr_) = pr_lin;
  *interp_->typed_tensor<float>(in_knob_ir_) = ir_lin;
  *interp_->typed_tensor<float>(in_knob_dw_) = dry_frac;

  // Helper to set optional scalar inputs by name
  auto setIf = [&](const char* name, float v){
    int idx = findInput(name);
    if (idx >= 0) { *interp_->typed_tensor<float>(idx) = v; }
  };

  // New: Output/Phase + Master EQ (all scalar [1,1])
  // Output gain: dB → linear
  const float out_lin = db2lin(gKnobs.outGain_dB.load());
  setIf("realtime_knob_outgain", out_lin);

  // Phase invert: 0 or 1
  setIf("realtime_knob_phaseinv", gKnobs.phaseInvert01.load());

  // Low cut (HP)
  setIf("realtime_knob_lc_hz",          gKnobs.lc_hz.load());
  setIf("realtime_knob_lc_slope_dbpoct",gKnobs.lc_slope_dbpoct.load());

  // Low shelf
  setIf("realtime_knob_ls_gain_db",     gKnobs.ls_gain_db.load());
  setIf("realtime_knob_ls_hz",          gKnobs.ls_hz.load());
  setIf("realtime_knob_ls_slope_dbpoct",gKnobs.ls_slope_dbpoct.load());

  // Mid
  setIf("realtime_knob_mid_gain_db",    gKnobs.mid_gain_db.load());
  setIf("realtime_knob_mid_hz",         gKnobs.mid_hz.load());
  setIf("realtime_knob_mid_q",          gKnobs.mid_q.load());

  // High shelf
  setIf("realtime_knob_hs_gain_db",     gKnobs.hs_gain_db.load());
  setIf("realtime_knob_hs_hz",          gKnobs.hs_hz.load());
  setIf("realtime_knob_hs_slope_dbpoct",gKnobs.hs_slope_dbpoct.load());

  // Feed state inputs (if streaming)
  if (isStreaming()) {
    for (auto &s : stateSlots_) {
      float* dst = interp_->typed_tensor<float>(s.input_idx);
      std::memcpy(dst, s.buf.data(), sizeof(float) * s.len());
    }
  }

  if (interp_->Invoke() != kTfLiteOk) {
    std::cerr << "[TFLite] Invoke failed\n";
    std::memset(outBlockT, 0, sizeof(float) * T_);
    styleProb = 0.0f;
    return;
  }

  // Audio
  const float* y = interp_->typed_tensor<float>(out_audio_);
  std::memcpy(outBlockT, y, sizeof(float) * T_);

  // Style (2D [1,1] preferred)
  styleProb = 0.0f;
  if (out_style_ >= 0) {
    auto* st = interp_->tensor(out_style_);
    if (st && st->type == kTfLiteFloat32) {
      if (st->dims->size == 2 && st->dims->data[0] == 1 && st->dims->data[1] == 1) {
        styleProb = *interp_->typed_tensor<float>(out_style_);
      } else {
        styleProb = interp_->typed_tensor<float>(out_style_)[0];
      }
    }
  }

  // Update states from packed bundle (if present)
  if (isStreaming() && out_bundle_ >= 0) {
    const TfLiteTensor* bt = interp_->tensor(out_bundle_);
    const float* bundle = interp_->typed_tensor<float>(out_bundle_);
    int total = 1;
    for (int i = 0; i < bt->dims->size; ++i) total *= bt->dims->data[i];
    if (out_splits_ >= 0) {
      const TfLiteTensor* sp = interp_->tensor(out_splits_);
      std::vector<int> lens;
      if (sp->type == kTfLiteInt32) {
        const int32_t* p = interp_->typed_tensor<int32_t>(out_splits_);
        int N = 1; for (int i=0;i<sp->dims->size;i++) N *= sp->dims->data[i];
        for (int i=0;i<N;i++) lens.push_back((int)p[i]);
      } else if (sp->type == kTfLiteInt64) {
        const int64_t* p = interp_->typed_tensor<int64_t>(out_splits_);
        int N = 1; for (int i=0;i<sp->dims->size;i++) N *= sp->dims->data[i];
        for (int i=0;i<N;i++) lens.push_back((int)p[i]);
      }
      size_t offset = 0;
      for (size_t i = 0; i < stateSlots_.size(); ++i) {
        int need = (i < lens.size()) ? lens[i] : stateSlots_[i].len();
        int have = std::max(0, std::min(need, total - (int)offset));
        if (have > 0) {
          std::memcpy(stateSlots_[i].buf.data(), bundle + offset, sizeof(float) * std::min(have, stateSlots_[i].len()));
        }
        offset += have;
      }
    } else {
      size_t offset = 0;
      for (auto &s : stateSlots_) {
        int have = std::max(0, std::min(s.len(), total - (int)offset));
        if (have > 0) {
          std::memcpy(s.buf.data(), bundle + offset, sizeof(float) * have);
        }
        offset += have;
      }
    }
  }
}

  // WINDOWED invoke (legacy non-streaming model): pass full [1,T,1] window.
  void invokeWindowed(const float* inWindowT, float* outWindowT, float &styleProb,
                    float pa_lin, float pr_lin, float ir_lin, float dry_frac,
                    float style_true, float style_force)
{
  float* pz = interp_->typed_tensor<float>(in_piezo_);
  std::memcpy(pz, inWindowT, sizeof(float) * T_);

  *interp_->typed_tensor<float>(in_st_true_)  = style_true;
  *interp_->typed_tensor<float>(in_st_force_) = style_force;

  *interp_->typed_tensor<float>(in_knob_pa_)  = pa_lin;
  *interp_->typed_tensor<float>(in_knob_pr_)  = pr_lin;
  *interp_->typed_tensor<float>(in_knob_ir_)  = ir_lin;
  *interp_->typed_tensor<float>(in_knob_dw_)  = dry_frac;

  auto setIf = [&](const char* name, float v){
    int idx = findInput(name);
    if (idx >= 0) { *interp_->typed_tensor<float>(idx) = v; }
  };

  const float out_lin = db2lin(gKnobs.outGain_dB.load());
  setIf("realtime_knob_outgain", out_lin);
  setIf("realtime_knob_phaseinv", gKnobs.phaseInvert01.load());

  setIf("realtime_knob_lc_hz",           gKnobs.lc_hz.load());
  setIf("realtime_knob_lc_slope_dbpoct", gKnobs.lc_slope_dbpoct.load());

  setIf("realtime_knob_ls_gain_db",      gKnobs.ls_gain_db.load());
  setIf("realtime_knob_ls_hz",           gKnobs.ls_hz.load());
  setIf("realtime_knob_ls_slope_dbpoct", gKnobs.ls_slope_dbpoct.load());

  setIf("realtime_knob_mid_gain_db",     gKnobs.mid_gain_db.load());
  setIf("realtime_knob_mid_hz",          gKnobs.mid_hz.load());
  setIf("realtime_knob_mid_q",           gKnobs.mid_q.load());

  setIf("realtime_knob_hs_gain_db",      gKnobs.hs_gain_db.load());
  setIf("realtime_knob_hs_hz",           gKnobs.hs_hz.load());
  setIf("realtime_knob_hs_slope_dbpoct", gKnobs.hs_slope_dbpoct.load());

  if (interp_->Invoke() != kTfLiteOk) {
    std::cerr << "[TFLite] Invoke failed\n";
    std::memset(outWindowT, 0, sizeof(float) * T_);
    styleProb = 0.0f;
    return;
  }
  const float* y = interp_->typed_tensor<float>(out_audio_);
  std::memcpy(outWindowT, y, sizeof(float) * T_);

  styleProb = 0.0f;
  if (out_style_ >= 0) {
    auto* st = interp_->tensor(out_style_);
    if (st && st->type == kTfLiteFloat32) {
      if (st->dims->size == 2 && st->dims->data[0] == 1 && st->dims->data[1] == 1) {
        styleProb = *interp_->typed_tensor<float>(out_style_);
      } else {
        styleProb = interp_->typed_tensor<float>(out_style_)[0];
      }
    }
  }
}

private:
  struct StateSlot {
    std::string name;
    int input_idx = -1;
    int L = 0;
    int C = 1;
    std::vector<float> buf;
    int len() const { return L * C; }
  };

  int findInput(const std::string& name) const {
    auto it = in_map_.find(name);
    if (it != in_map_.end()) return it->second;
    for (auto &kv : in_map_) {
      if (kv.first == name) return kv.second;
    }
    return -1;
  }

  void discoverStateInputs() {
    // Collect realtime_state_* inputs and sort them lexicographically by name
    std::vector<StateSlot> tmp;
    for (auto &kv : in_map_) {
      const std::string& nm = kv.first;
      if (nm.rfind("realtime_state_", 0) == 0) {
        int idx = kv.second;
        const TfLiteTensor* t = interp_->tensor(idx);
        if (!t || t->dims->size < 2) continue;
        StateSlot s;
        s.name = nm;
        s.input_idx = idx;
        // Expect shape [1, L, C] or [1, L]
        if (t->dims->size == 3) {
          s.L = t->dims->data[1];
          s.C = t->dims->data[2];
        } else if (t->dims->size == 2) {
          s.L = t->dims->data[1];
          s.C = 1;
        } else {
          continue;
        }
        s.buf.resize(std::max(1, s.L*s.C), 0.0f);
        tmp.push_back(std::move(s));
      }
    }
    std::sort(tmp.begin(), tmp.end(), [](const StateSlot& a, const StateSlot& b){
      return a.name < b.name;
    });
    stateSlots_.swap(tmp);
  }

  void classifyOutputsByShape() {
  out_audio_  = -1;
  out_style_  = -1;
  out_bundle_ = -1;
  out_splits_ = -1;

  size_t bestAudioTotal  = 0;
  size_t bestBundleTotal = 0;

  // 1) Audio = 3D float (prefer [1,T,1]); pick the largest 3D if multiple
  for (int idx : interp_->outputs()) {
    TfLiteTensor* t = interp_->tensor(idx);
    if (!t || !t->dims) continue;
    if (t->type == kTfLiteFloat32 && t->dims->size == 3) {
      size_t total = 1;
      for (int i = 0; i < t->dims->size; ++i) total *= t->dims->data[i];
      if (total > bestAudioTotal) {
        bestAudioTotal = total;
        out_audio_ = idx;
      }
    }
  }

  // 2) Style = 2D float (1,1) if present
  for (int idx : interp_->outputs()) {
    TfLiteTensor* t = interp_->tensor(idx);
    if (!t || !t->dims) continue;
    if (t->type == kTfLiteFloat32 && t->dims->size == 2 &&
        t->dims->data[0] == 1 && t->dims->data[1] == 1) {
      out_style_ = idx;
      break;
    }
  }

  // 3) Bundle = largest 2D float (excluding (1,1))
  for (int idx : interp_->outputs()) {
    TfLiteTensor* t = interp_->tensor(idx);
    if (!t || !t->dims) continue;
    if (t->type == kTfLiteFloat32 && t->dims->size == 2) {
      int r = t->dims->data[0], c = t->dims->data[1];
      if (!(r == 1 && c == 1)) {
        size_t total = static_cast<size_t>(r) * static_cast<size_t>(c);
        if (total > bestBundleTotal) {
          bestBundleTotal = total;
          out_bundle_ = idx;
        }
      }
    }
  }

  // 4) Splits = int vector (1,Nstates) or (Nstates,)
  const int nStates = static_cast<int>(stateSlots_.size());
  for (int idx : interp_->outputs()) {
    TfLiteTensor* t = interp_->tensor(idx);
    if (!t || !t->dims) continue;
    if (!isIntType(t->type)) continue;
    int dims = t->dims->size;
    if (dims == 2 && t->dims->data[0] == 1 && t->dims->data[1] == nStates) {
      out_splits_ = idx;
      break;
    }
    if (dims == 1 && t->dims->data[0] == nStates) {
      out_splits_ = idx;
      break;
    }
  }

  // Fallbacks if something still missing:
  if (out_audio_ < 0) {
    // pick any 3D float
    for (int idx : interp_->outputs()) {
      TfLiteTensor* t = interp_->tensor(idx);
      if (t && t->type == kTfLiteFloat32 && t->dims && t->dims->size == 3) {
        out_audio_ = idx;
        break;
      }
    }
    if (out_audio_ < 0) {
      // as a last resort, choose the largest tensor
      size_t best = 0;
      for (int idx : interp_->outputs()) {
        TfLiteTensor* t = interp_->tensor(idx);
        if (!t || !t->dims) continue;
        size_t total = 1;
        for (int i = 0; i < t->dims->size; ++i) total *= t->dims->data[i];
        if (total > best) { best = total; out_audio_ = idx; }
      }
    }
  }
  if (out_style_ < 0) {
    // choose any small 2D/1D float tensor
    for (int idx : interp_->outputs()) {
      TfLiteTensor* t = interp_->tensor(idx);
      if (t && t->type == kTfLiteFloat32 && t->dims && t->dims->size <= 2) {
        out_style_ = idx;
        break;
      }
    }
  }
}

  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter>     interp_;
  #ifdef TFLITE_WITH_XNNPACK
    TfLiteDelegate* xnnpack_ = nullptr;
  #else
    void* xnnpack_ = nullptr;
  #endif

  std::unordered_map<std::string, int> in_map_, out_map_;
  int in_piezo_=-1, in_st_true_=-1, in_st_force_=-1, in_knob_pa_=-1, in_knob_pr_=-1, in_knob_ir_=-1, in_knob_dw_=-1;
  int out_audio_=-1, out_style_=-1, out_bundle_=-1, out_splits_=-1;
  int T_ = 0;

  std::vector<StateSlot> stateSlots_;
};

// ───────────────────────── INFERENCE THREAD ─────────────────────────
struct InferCtx {
  Shared* sh;
  int hop;
  int sampleRate;
  std::string modelPath;
};

void inferenceThread(InferCtx ctx) {
#if defined(__APPLE__)
  #ifdef QOS_CLASS_AUDIO
    pthread_set_qos_class_self_np(QOS_CLASS_AUDIO, 0);
  #else
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  #endif
  struct sched_param sp;
  sp.sched_priority = sched_get_priority_max(SCHED_FIFO);
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
#endif

  TFLiteRealtime rt(ctx.modelPath);
  const int T   = rt.T();
  const int hop = ctx.hop;

  if (hop > T) {
    std::cerr << "[inference] Error: blocksize/hop ("<<hop<<") must be <= T ("<<T<<")\n";
    gStopFlag = true; return;
  }

  // Make sure we run with hop == T for best behavior (classifier looks over full T)
  if (hop != T) {
    std::cerr << "[inference] WARN: hop ("<<hop<<") != T ("<<T<<"). Consider --blocksize "<<T<<" for best style stability.\n";
  }

  uint64_t lastBypassEpoch = gBypassEpoch.load(std::memory_order_acquire);
  std::vector<float> xhop(hop, 0.0f);
  std::vector<float> yT(T, 0.0f);
  float styleP = 0.0f;  // raw per-hop style tensor

  size_t hopsInWindow = std::max<size_t>(1, (size_t)std::ceil(windowSec * double(ctx.sampleRate) / double(hop)));
  ProbWindow probWin;
  probWin.init(nextPow2(hopsInWindow + 8));

  // Stabilized display probability = blend(avg_5s, current_ema)
  const float kStabBlend = 0.7f;  // larger → more weight to long window
  float stable_for_next  = 0.5f;  // used if guided-auto feeds model (one-hop lag)

  // Track last printed knob values
  float prev_pa  = gKnobs.pickAttack_dB.load();
  float prev_pr  = gKnobs.pickReso_dB.load();
  float prev_ir  = gKnobs.ir_dB.load();
  float prev_wet = gKnobs.wet_pct.load();
  int   prev_mode= gStyleMode.load();

  auto pickOrFinger = [&](float p, int styleMode)->const char* {
    if (styleMode == static_cast<int>(StyleMode::ForcePick))   return "PICK (forced)";
    if (styleMode == static_cast<int>(StyleMode::ForceFinger)) return "FINGER (forced)";
    return (p >= 0.60f) ? "PICK" : "FINGER"; // slightly conservative UI threshold
  };

  auto printKnobStatus = [&](const char* tag, float dispP){
    int sm = gStyleMode.load();
    std::cout << (tag ? tag : "[UI]") << " PA=" << gKnobs.pickAttack_dB.load() << " dB  "
              << "PR=" << gKnobs.pickReso_dB.load()   << " dB  "
              << "IR=" << gKnobs.ir_dB.load()         << " dB  "
              << "Wet="<< gKnobs.wet_pct.load()       << " %  "
              << "Pred P=" << dispP << "  "
              << "Pred=" << pickOrFinger(dispP, sm) << "  "
              << "Mode=" << styleModeName(sm)
              << (gGuidedAuto.load() ? "  (Guided AUTO ON)" : "")
              << "\n";
  };

  auto checkAndReportKnobChanges = [&](float dispP){
    float pa  = gKnobs.pickAttack_dB.load();
    float pr  = gKnobs.pickReso_dB.load();
    float ir  = gKnobs.ir_dB.load();
    float wet = gKnobs.wet_pct.load();
    int   sm  = gStyleMode.load();
    if (pa != prev_pa || pr != prev_pr || ir != prev_ir || wet != prev_wet || sm != prev_mode) {
      printKnobStatus("[UI]", dispP);
      prev_pa = pa; prev_pr = pr; prev_ir = ir; prev_wet = wet; prev_mode = sm;
    }
  };

  // Wait for at least one hop of input
  while (!gStopFlag && ctx.sh->in.available() < (size_t)hop)
    std::this_thread::yield();
  if (gStopFlag) return;

  const bool streaming = rt.isStreaming();
  if (streaming) {
    rt.resetStates();
  }

  auto onBypassEdgeIfAny = [&](){
    uint64_t cur = gBypassEpoch.load(std::memory_order_acquire);
    if (cur != lastBypassEpoch) {
      lastBypassEpoch = cur;
      if (streaming) {
        rt.resetStates();
      }
      probWin.reset();
      // Drop any queued processed audio to be extra safe
      ctx.sh->out.r.store(ctx.sh->out.w.load(std::memory_order_acquire), std::memory_order_release);
    }
  };

  // Initial hop
  size_t got = ctx.sh->in.pop(xhop.data(), hop);
  if (got < (size_t)hop) std::fill(xhop.begin()+got, xhop.end(), 0.0f);

  // Helpers to call TFLite for one block
  auto do_invoke_streaming = [&](const float* coreBuf, int coreLen, float guidedStableForAuto){
    float pa_lin = std::pow(10.0f, gKnobs.pickAttack_dB.load() / 20.0f);
    float pr_lin = std::pow(10.0f, gKnobs.pickReso_dB.load()   / 20.0f);
    float ir_lin = std::pow(10.0f, gKnobs.ir_dB.load()         / 20.0f);
    float wet    = std::max(0.0f, std::min(100.0f, gKnobs.wet_pct.load()));
    float dryFrac= 1.0f - (wet / 100.0f);

    int sm = gStyleMode.load();
    float st_true = 0.0f, st_force = 0.0f;
    if (sm == static_cast<int>(StyleMode::ForcePick))   { st_true = 1.0f; st_force = 1.0f; }
    if (sm == static_cast<int>(StyleMode::ForceFinger)) { st_true = 0.0f; st_force = 1.0f; }
    if (sm == static_cast<int>(StyleMode::Auto) && gGuidedAuto.load()) {
      st_true  = std::max(0.0f, std::min(1.0f, guidedStableForAuto));
      st_force = 1.0f; // override model’s internal style with 5s-stabilized prob
    }

    rt.invokeStreamingBlock(coreBuf, coreLen, yT.data(), styleP, pa_lin, pr_lin, ir_lin, dryFrac, st_true, st_force);
  };

  auto do_invoke_windowed = [&](const float* windowT, float guidedStableForAuto){
    float pa_lin = std::pow(10.0f, gKnobs.pickAttack_dB.load() / 20.0f);
    float pr_lin = std::pow(10.0f, gKnobs.pickReso_dB.load()   / 20.0f);
    float ir_lin = std::pow(10.0f, gKnobs.ir_dB.load()         / 20.0f);
    float wet    = std::max(0.0f, std::min(100.0f, gKnobs.wet_pct.load()));
    float dryFrac= 1.0f - (wet / 100.0f);

    int sm = gStyleMode.load();
    float st_true = 0.0f, st_force = 0.0f;
    if (sm == static_cast<int>(StyleMode::ForcePick))   { st_true = 1.0f; st_force = 1.0f; }
    if (sm == static_cast<int>(StyleMode::ForceFinger)) { st_true = 0.0f; st_force = 1.0f; }
    if (sm == static_cast<int>(StyleMode::Auto) && gGuidedAuto.load()) {
      st_true  = std::max(0.0f, std::min(1.0f, guidedStableForAuto));
      st_force = 1.0f;
    }

    rt.invokeWindowed(windowT, yT.data(), styleP, pa_lin, pr_lin, ir_lin, dryFrac, st_true, st_force);
  };

  // First block
  if (streaming) {
    do_invoke_streaming(xhop.data(), hop, stable_for_next);
    ctx.sh->out.push(yT.data(), hop);
    // Update 5s window with EMA (or raw) from this hop
    float emaP = rt.getStyleEma();
    float pHop = std::isfinite(emaP) ? emaP : styleP;
    probWin.push(pHop, float(hop));
    float pAvgWindowSec = probWin.mean();
    float dispP = (1.0f - kStabBlend) * pHop + kStabBlend * pAvgWindowSec;
    stable_for_next = dispP;
    checkAndReportKnobChanges(dispP);
  } else {
    // Legacy windowed: left = T - hop, right=0
    const int left = T - hop;
    std::vector<float> win(T, 0.0f);
    std::memcpy(win.data() + left, xhop.data(), sizeof(float)*hop);
    do_invoke_windowed(win.data(), stable_for_next);
    ctx.sh->out.push(yT.data() + left, hop);
    float emaP = rt.getStyleEma();
    float pHop = std::isfinite(emaP) ? emaP : styleP;
    probWin.push(pHop, float(hop));
    float pAvgWindowSec = probWin.mean();
    float dispP = (1.0f - kStabBlend) * pHop + kStabBlend * pAvgWindowSec;
    stable_for_next = dispP;
    checkAndReportKnobChanges(dispP);
  }

  // Main loop
  if (streaming) {
    while (!gStopFlag) {
      while (!gStopFlag && ctx.sh->in.available() < (size_t)hop)
        std::this_thread::yield();
      if (gStopFlag) break;

      size_t got2 = ctx.sh->in.pop(xhop.data(), hop);
      if (got2 < (size_t)hop) std::fill(xhop.begin()+got2, xhop.end(), 0.0f);

      // Reset state/rings on bypass edge
      onBypassEdgeIfAny();

      auto t0 = std::chrono::steady_clock::now();

      // Use the previous stabilized prob for guided AUTO (one-hop lag)
      do_invoke_streaming(xhop.data(), hop, stable_for_next);
      ctx.sh->out.push(yT.data(), hop);

      // Update rolling {windowSec}s window and stabilized display prob
      float emaP = rt.getStyleEma();
      float pHop = std::isfinite(emaP) ? emaP : styleP;
      probWin.push(pHop, float(hop));
      float pAvgWindowSec = probWin.mean();
      float dispP = (1.0f - kStabBlend) * pHop + kStabBlend * pAvgWindowSec;
      stable_for_next = dispP;

      auto t1 = std::chrono::steady_clock::now();
      checkAndReportKnobChanges(dispP);

      double proc_ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
      static auto   last_report = t1;
      static double sum_proc = 0.0;
      static size_t cnt = 0;
      sum_proc += proc_ms; ++cnt;
      if (t1 - last_report >= std::chrono::seconds(PROCESSING_LOGGING_WAIT_SEC)) {
        double avg_proc = sum_proc / double(cnt);
        double io_ms    = (gIOInputLatencySec + gIOOutputLatencySec)*1000.0;
        double total_ms = avg_proc + io_ms;
        int sm = gStyleMode.load();
        std::cerr << "[inference/stream] Proc="<<avg_proc<<" ms  I/O="<<io_ms<<" ms  Total~"<<total_ms<<" ms"
                  << "  Pred" << windowSec << "s=" << pAvgWindowSec << "  DispP=" << dispP
                  << "  pred="<<pickOrFinger(dispP, sm)<<"  mode="<<styleModeName(sm)
                  << (gGuidedAuto.load() ? "  (Guided)" : "")
                  << "\n";
        last_report = t1; sum_proc = 0.0; cnt = 0;
      }
    }
  } else {
    const int left = T - hop;
    std::vector<float> win(T, 0.0f);
    while (!gStopFlag) {
      while (!gStopFlag && ctx.sh->in.available() < (size_t)hop)
        std::this_thread::yield();
      if (gStopFlag) break;

      size_t got2 = ctx.sh->in.pop(xhop.data(), hop);
      if (got2 < (size_t)hop) std::fill(xhop.begin()+got2, xhop.end(), 0.0f);

      // Reset state/rings on bypass edge
      onBypassEdgeIfAny();

      auto t0 = std::chrono::steady_clock::now();

      // Slide window left by hop; append new hop at end
      if (left > 0) std::memmove(win.data(), win.data()+hop, sizeof(float)*(T-hop));
      std::memcpy(win.data() + (T - hop), xhop.data(), sizeof(float)*hop);

      do_invoke_windowed(win.data(), stable_for_next);
      ctx.sh->out.push(yT.data() + (T - hop), hop);

      float emaP = rt.getStyleEma();
      float pHop = std::isfinite(emaP) ? emaP : styleP;
      probWin.push(pHop, float(hop));
      float pAvgWindowSec = probWin.mean();
      float dispP = (1.0f - kStabBlend) * pHop + kStabBlend * pAvgWindowSec;
      stable_for_next = dispP;

      auto t1 = std::chrono::steady_clock::now();
      checkAndReportKnobChanges(dispP);

      double proc_ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
      static auto   last_report = t1;
      static double sum_proc = 0.0;
      static size_t cnt = 0;
      sum_proc += proc_ms; ++cnt;
      if (t1 - last_report >= std::chrono::seconds(PROCESSING_LOGGING_WAIT_SEC)) {
        double avg_proc = sum_proc / double(cnt);
        double io_ms    = (gIOInputLatencySec + gIOOutputLatencySec)*1000.0;
        double total_ms = avg_proc + io_ms;
        int sm = gStyleMode.load();
        std::cerr << "[inference/window] Proc="<<avg_proc<<" ms  I/O="<<io_ms<<" ms  Total~"<<total_ms<<" ms"
                  << "  Pred" << windowSec << "s=" << pAvgWindowSec << "  DispP=" << dispP
                  << "  pred="<<pickOrFinger(dispP, sm)<<"  mode="<<styleModeName(sm)
                  << (gGuidedAuto.load() ? "  (Guided)" : "")
                  << "\n";
        last_report = t1; sum_proc = 0.0; cnt = 0;
      }
    }
  }
}

// ───────────────────────── PA CALLBACK ─────────────────────────

static int paCallback(const void *inputBuffer,
                      void *outputBuffer,
                      unsigned long framesPerBuffer,
                      const PaStreamCallbackTimeInfo *timeInfo,
                      PaStreamCallbackFlags,
                      void *userData)
{
  auto *sh = static_cast<Shared*>(userData);
  auto *in = static_cast<const float*>(inputBuffer);
  auto *out= static_cast<float*>(outputBuffer);
  size_t hop = framesPerBuffer;

  // Update measured E2E latency using PA's hardware timestamps.
  if (timeInfo) {
    static double s_prevInputAdcTime = 0.0;
    s_prevInputAdcTime = timeInfo->inputBufferAdcTime;

    // Bypass edge detection: drop all queued audio immediately on any toggle.
    static uint64_t s_lastBypassEpoch = 0;
    static bool s_lastBypass = false;
    uint64_t curEpoch = gBypassEpoch.load(std::memory_order_acquire);
    bool curBypass = gBypass.load(std::memory_order_relaxed);
    if (curEpoch != s_lastBypassEpoch || curBypass != s_lastBypass) {
      sh->in.r.store(sh->in.w.load(std::memory_order_acquire), std::memory_order_release);
      sh->out.r.store(sh->out.w.load(std::memory_order_acquire), std::memory_order_release);
      // Reset measurement to avoid reporting garbage around toggle
      s_prevInputAdcTime = 0.0;
      s_lastBypassEpoch = curEpoch;
      s_lastBypass = curBypass;
    }
  }

  if (in) sh->in.push(in, hop);

  if (gBypass.load(std::memory_order_relaxed)) {
    // While bypass is ON, ensure the processed output ring stays empty
    sh->out.r.store(sh->out.w.load(std::memory_order_acquire), std::memory_order_release);

    if (in) {
      for (size_t i=0;i<hop;i++){
        float v = in[i];
        out[2*i] = v; out[2*i+1] = v;
      }
    } else {
      std::memset(out, 0, sizeof(float)*2*hop);
    }
    return paContinue;
  }

  float *temp = sh->tempOut.data();
  size_t got  = sh->out.pop(temp, hop);
  if (got < hop) std::memset(temp+got, 0, sizeof(float)*(hop-got));
  for (size_t i=0;i<hop;i++){
    float v = temp[i];
    out[2*i] = v; out[2*i+1] = v;
  }
  return paContinue;
}

// ───────────────────────── MAC HAL BUFFER SIZE ─────────────────────────
#ifdef __APPLE__
static void setMacHardwareBufferSize(const DeviceInfo& paDev, int blockSize, bool isInput) {
  AudioObjectID coreAudioDeviceID = kAudioObjectUnknown;

  // Enumerate devices to match by name
  AudioObjectPropertyAddress propAddr = {
    kAudioHardwarePropertyDevices,
    kAudioObjectPropertyScopeGlobal,
    kAudioObjectPropertyElementMain
  };
  UInt32 dataSize = 0;
  OSStatus deviceErr = AudioObjectGetPropertyDataSize(
      kAudioObjectSystemObject, &propAddr, 0, nullptr, &dataSize);
  if (deviceErr) { std::cerr << "Error get dev size: " << deviceErr << "\n"; return; }

  int deviceCount = dataSize / sizeof(AudioDeviceID);
  std::vector<AudioDeviceID> devs(deviceCount);
  deviceErr = AudioObjectGetPropertyData(
      kAudioObjectSystemObject, &propAddr, 0, nullptr, &dataSize, devs.data());
  if (deviceErr) { std::cerr << "Error get dev list: " << deviceErr << "\n"; return; }

  for (auto devID : devs) {
    AudioObjectPropertyAddress nameAddr = {
      kAudioObjectPropertyName,
      kAudioObjectPropertyScopeGlobal,
      kAudioObjectPropertyElementMain
    };
    CFStringRef cfName = nullptr;
    UInt32 nSize = sizeof(cfName);
    OSStatus err = AudioObjectGetPropertyData(
        devID, &nameAddr, 0, nullptr, &nSize, &cfName);
    if (err == noErr && cfName) {
      char nameBuf[256] = {0};
      CFStringGetCString(cfName, nameBuf, sizeof(nameBuf), kCFStringEncodingUTF8);
      CFRelease(cfName);
      if (std::string(nameBuf) == paDev.name) {
        coreAudioDeviceID = devID; break;
      }
    }
  }
  if (coreAudioDeviceID == kAudioObjectUnknown) {
    std::cerr << "[HAL] could not find CoreAudio device by name: " << paDev.name << "\n";
    return;
  }

  // Set buffer frame size
  UInt32 newBufferSize = (UInt32)blockSize;
  AudioObjectPropertyAddress bufAddr = {
    kAudioDevicePropertyBufferFrameSize,
    kAudioObjectPropertyScopeGlobal,
    kAudioObjectPropertyElementMain
  };
  deviceErr = AudioObjectSetPropertyData(
      coreAudioDeviceID, &bufAddr, 0, nullptr, sizeof(newBufferSize), &newBufferSize);
  if (deviceErr) {
    std::cerr << "Error setting buffer size: " << deviceErr << "\n";
  }
  // Read back
  UInt32 actualSize = 0; UInt32 asize = sizeof(actualSize);
  deviceErr = AudioObjectGetPropertyData(coreAudioDeviceID, &bufAddr, 0, nullptr, &asize, &actualSize);
  if (!deviceErr) {
    std::cout << "[HAL] " << (isInput?"Input":"Output") << " buffer now: " << actualSize << " frames\n";
  }
}
#endif

// ───────────────────────── MAIN ─────────────────────────
int main(int argc, char** argv) {
  int  blockSize = 32; // must be <= T
  int  inDevId = -1, outDevId = -1;
  std::string tflitePath;

  bool listDev = false;
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"--list-devices")) listDev = true;
    else if(!strcmp(argv[i],"--blocksize") && i+1<argc) blockSize = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--input-device") && i+1<argc) inDevId  = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--output-device")&&i+1<argc) outDevId = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--tflite") && i+1<argc) tflitePath = argv[++i];
    else {
      std::cerr<<"Unknown arg "<<argv[i]<<"\n";
      return 1;
    }
  }

  auto devices = getDevices();
  if (listDev) { listDevices(devices); return 0; }

  if (inDevId<0 || outDevId<0 || tflitePath.empty()) {
    std::cerr << "Usage: --tflite <model.tflite> --input-device <id> --output-device <id> [--blocksize N]\n";
    listDevices(devices);
    return 1;
  }

  DeviceInfo inDev{}, outDev{};
  for (auto &d : devices) {
    if (d.devId == inDevId)  inDev = d;
    if (d.devId == outDevId) outDev = d;
  }
  if (inDev.devId<0 || outDev.devId<0) {
    std::cerr<<"Invalid device IDs\n";
    listDevices(devices);
    return 1;
  }

#ifdef __APPLE__
  setMacHardwareBufferSize(inDev,  blockSize, true);
  setMacHardwareBufferSize(outDev, blockSize, false);
#endif

  PaError err = Pa_Initialize();
  if (err != paNoError) {
    std::cerr<<"Pa_Initialize failed: "<<Pa_GetErrorText(err)<<"\n";
    return 1;
  }
  auto *diIn  = Pa_GetDeviceInfo(inDevId);
  auto *diOut = Pa_GetDeviceInfo(outDevId);
  int sampleRate = (int)std::lround(diIn->defaultSampleRate);

  std::cout<<"[MAIN] In ["<<inDevId<<"] "<<diIn->name<<" sr="<<diIn->defaultSampleRate<<"\n";
  std::cout<<"[MAIN] Out["<<outDevId<<"] "<<diOut->name<<" sr="<<diOut->defaultSampleRate<<"\n";
  std::cout<<"[MAIN] blockSize="<<blockSize<<"\n";
  std::cout<<"[MAIN] model: "<<tflitePath<<"\n";

  // PortAudio stream params
  double wantLatencySecs = double(blockSize) / sampleRate;
  PaStreamParameters ip = { inDevId, 1, paFloat32, wantLatencySecs, nullptr };
  PaStreamParameters op = { outDevId, 2, paFloat32, wantLatencySecs, nullptr };

  // stdin non-blocking for UI
  termios oldt,newt;
  tcgetattr(STDIN_FILENO,&oldt);
  newt = oldt; newt.c_lflag &= ~(ICANON|ECHO);
  tcsetattr(STDIN_FILENO,TCSANOW,&newt);
  fcntl(STDIN_FILENO,F_SETFL,O_NONBLOCK);

  std::signal(SIGINT, signalHandler);

  // Shared rings
  size_t cap = 1<<14;
  Shared shared(cap);
  shared.tempOut.resize(blockSize);

  // Spawn inference thread
  InferCtx ictx;
  ictx.sh = &shared; ictx.hop = blockSize; ictx.sampleRate = sampleRate; ictx.modelPath = tflitePath;
  std::thread infThread([&](){ inferenceThread(ictx); });

  // Keyboard thread for bypass + knobs
  std::thread kbThread([&](){

  std::cout
    << "[MAIN] Keys: "
    << "b=Bypass  "
    << "q/a=PA +/-1dB  "
    << "w/s=PR +/-1dB  "
    << "e/d=IR +/-1dB  "
    << "r/f=Wet +/-5%  "
    << "t=Style mode (auto→pick→finger)  "
    << "g=Guided AUTO on/off  "
    << "p=print"
    << "\n";

    while (!gStopFlag) {
      int c = getchar();
      if (c == EOF) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); continue; }
      if (c=='b'||c=='B') {
        bool now = !gBypass.load(std::memory_order_relaxed);
        bool prev = gBypass.exchange(now, std::memory_order_acq_rel);
        gBypassEpoch.fetch_add(1, std::memory_order_acq_rel);
        // Drop all queued audio immediately
        shared.in.r.store(shared.in.w.load(std::memory_order_acquire), std::memory_order_release);
        shared.out.r.store(shared.out.w.load(std::memory_order_acquire), std::memory_order_release);
        std::cout<<"[UI] Bypass "<<(now?"ON":"OFF")<<"\n";
      } else if (c=='q') { gKnobs.pickAttack_dB.store( clampf(gKnobs.pickAttack_dB.load()+1.0f, -maxKnobDbChange, maxKnobDbChange) ); }
        else if (c=='a') { gKnobs.pickAttack_dB.store( clampf(gKnobs.pickAttack_dB.load()-1.0f, -maxKnobDbChange, maxKnobDbChange) ); }
        else if (c=='w') { gKnobs.pickReso_dB.store( clampf(gKnobs.pickReso_dB.load()+1.0f, -maxKnobDbChange, maxKnobDbChange) ); }
        else if (c=='s') { gKnobs.pickReso_dB.store( clampf(gKnobs.pickReso_dB.load()-1.0f, -maxKnobDbChange, maxKnobDbChange) ); }
        else if (c=='e') { gKnobs.ir_dB.store( clampf(gKnobs.ir_dB.load()+1.0f, -maxKnobDbChange, maxKnobDbChange) ); }
        else if (c=='d') { gKnobs.ir_dB.store( clampf(gKnobs.ir_dB.load()-1.0f, -maxKnobDbChange, maxKnobDbChange) ); }
        else if (c=='r') { gKnobs.wet_pct.store( clampf(gKnobs.wet_pct.load()+5.0f, 0.0f, 100.0f) ); }
        else if (c=='f') { gKnobs.wet_pct.store( clampf(gKnobs.wet_pct.load()-5.0f, 0.0f, 100.0f) ); }
        else if (c=='t') {
          int cur = gStyleMode.load();
          int next = (cur == static_cast<int>(StyleMode::Auto)) ? static_cast<int>(StyleMode::ForcePick)
                     : (cur == static_cast<int>(StyleMode::ForcePick)) ? static_cast<int>(StyleMode::ForceFinger)
                     : static_cast<int>(StyleMode::Auto);
          gStyleMode.store(next);
          std::cout<<"[UI] Style mode = "<<styleModeName(next)<<"\n";
        }
        else if (c=='p') {
          std::cout<<"[UI] PA="<<gKnobs.pickAttack_dB.load()<<" dB  "
                   <<"PR="<<gKnobs.pickReso_dB.load()<<" dB  "
                   <<"IR="<<gKnobs.ir_dB.load()<<" dB  "
                   <<"Wet="<<gKnobs.wet_pct.load()<<" %  "
                   <<"Style="<<styleModeName(gStyleMode.load())<<"\n";
        }

        else if (c=='g' || c=='G') {
          bool now = !gGuidedAuto.load();
          gGuidedAuto.store(now);
          std::cout << "[UI] Guided AUTO style " << (now ? "ON (model style overridden by " + std::to_string(windowSec) + "s-stabilized prob)" : "OFF") << "\n";
        }
    }
  });

  StartKnobWebUI(8088);

  PaStream *stream = nullptr;
  err = Pa_OpenStream(&stream, &ip, &op, sampleRate, (unsigned long)blockSize, paNoFlag, paCallback, &shared);
  if (err != paNoError) {
    std::cerr<<"Pa_OpenStream failed: "<<Pa_GetErrorText(err)<<"\n";
    Pa_Terminate(); gStopFlag = true;
    infThread.join(); kbThread.join();
    tcsetattr(STDIN_FILENO,TCSANOW,&oldt);
    return 1;
  }
  err = Pa_StartStream(stream);
  if (err != paNoError) {
    std::cerr<<"Pa_StartStream failed: "<<Pa_GetErrorText(err)<<"\n";
    Pa_CloseStream(stream); Pa_Terminate(); gStopFlag = true;
    infThread.join(); kbThread.join();
    tcsetattr(STDIN_FILENO,TCSANOW,&oldt);
    return 1;
  }

  auto *info = Pa_GetStreamInfo(stream);
  gIOInputLatencySec  = info->inputLatency;
  gIOOutputLatencySec = info->outputLatency;
  std::cerr<<"[MAIN] Actual I/O buffer: in="<<(int)std::round(gIOInputLatencySec*sampleRate)
           <<" out="<<(int)std::round(gIOOutputLatencySec*sampleRate)<<" frames\n";

  std::cout<<"[MAIN] Streaming… (Ctrl-C to quit)\n";
  while (!gStopFlag) Pa_Sleep(100);

  std::cout<<"[MAIN] Stopping…\n";

  StopKnobWebUI();

  Pa_StopStream(stream);
  Pa_CloseStream(stream);
  Pa_Terminate();

  infThread.join();
  kbThread.join();
  tcsetattr(STDIN_FILENO,TCSANOW,&oldt);
  std::cout<<"[MAIN] Done.\n";
  return 0;
}