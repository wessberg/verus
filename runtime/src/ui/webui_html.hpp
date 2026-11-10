#pragma once
// The web UI page (HTML + JS) as one string.

#include "common.hpp"

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
  .row { display:flex; gap:16px; flex-wrap:wrap; align-items:center; justify-content:center; margin-block-end: 14px; }
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
      <canvas id="knob_main" width="110" height="110"></canvas>
      <div class="label">Main</div>
      <div class="value"><span id="val_main">__DEF_MAIN_DB__</span> dB</div>
    </div>
    <div class="knob">
      <canvas id="knob_pa" width="110" height="110"></canvas>
      <div class="label">Pick Attack</div>
      <div class="value"><span id="val_pa">__DEF_PA_DB__</span> dB</div>
    </div>
    <div class="knob">
      <canvas id="knob_pr" width="110" height="110"></canvas>
      <div class="label">Pick Resonance</div>
      <div class="value"><span id="val_pr">__DEF_PR_DB__</span> dB</div>
    </div>
    <div class="knob">
      <canvas id="knob_ir" width="110" height="110"></canvas>
      <div class="label">IR Gain</div>
      <div class="value"><span id="val_ir">__DEF_IR_DB__</span> dB</div>
    </div>
    <div class="knob">
      <canvas id="knob_wet" width="110" height="110"></canvas>
      <div class="label">Wet</div>
      <div class="value"><span id="val_wet">__DEF_WET_PCT__</span> %</div>
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
          <div class="value"><span id="val_outg">__DEF_OUTG_DB__</span> dB</div>
        </div>
        <div class="panel" style="display:flex;align-items:center;justify-content:center;gap:10px;min-width:200px;">
          <div class="btn" id="btnPhase">Phase Invert: __DEF_PHASE_TEXT__</div>
        </div>
      </div>
    </div>
  </div>

    <!-- Feedback Protection -->
  <div class="grid">
    <div class="panel">
      <div class="row">
        <div class="title">Feedback Protection</div>
        <div class="btn" id="btnAFB">Feedback Prot: __DEF_AFB_ENABLED_TEXT__</div>
      </div>
      <div class="hrow">
        <div class="knob">
          <canvas id="knob_afb_shift" width="110" height="110"></canvas>
          <div class="label">Shift</div>
          <div class="value"><span id="val_afb_shift">__DEF_AFB_SHIFT_HZ__</span> Hz</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Noise Gate -->
  <div class="grid">
    <div class="panel">
      <div class="row">
        <div class="title">Noise Gate</div>
        <div class="btn" id="btnGate">Gate: __DEF_GATE_ENABLED_TEXT__</div>
      </div>
      <div class="hrow">
        <div class="knob">
          <canvas id="knob_gate_thresh" width="110" height="110"></canvas>
          <div class="label">Threshold</div>
          <div class="value"><span id="val_gate_thresh">__DEF_GATE_THRESH_DB__</span> dB</div>
        </div>
        <div class="knob">
          <canvas id="knob_gate_hyst" width="110" height="110"></canvas>
          <div class="label">Hysteresis</div>
          <div class="value"><span id="val_gate_hyst">__DEF_GATE_HYST_DB__</span> dB</div>
        </div>
        <div class="knob">
          <canvas id="knob_gate_range" width="110" height="110"></canvas>
          <div class="label">Range</div>
          <div class="value"><span id="val_gate_range">__DEF_GATE_RANGE_DB__</span> dB</div>
        </div>
      </div>
      <div class="hrow">
        <div class="knob">
          <canvas id="knob_gate_attack" width="110" height="110"></canvas>
          <div class="label">Attack</div>
          <div class="value"><span id="val_gate_attack">__DEF_GATE_ATTACK_MS__</span> ms</div>
        </div>
        <div class="knob">
          <canvas id="knob_gate_release" width="110" height="110"></canvas>
          <div class="label">Release</div>
          <div class="value"><span id="val_gate_release">__DEF_GATE_RELEASE_MS__</span> ms</div>
        </div>
        <div class="knob">
          <canvas id="knob_gate_hold" width="110" height="110"></canvas>
          <div class="label">Hold</div>
          <div class="value"><span id="val_gate_hold">__DEF_GATE_HOLD_MS__</span> ms</div>
        </div>
        <div class="knob">
          <canvas id="knob_gate_hpf" width="110" height="110"></canvas>
          <div class="label">SC HPF</div>
          <div class="value"><span id="val_gate_hpf">__DEF_GATE_HPF_HZ__</span> Hz</div>
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
          <div class="value"><span id="val_lc_hz">__DEF_LC_HZ__</span> Hz</div>
        </div>
        <div class="knob">
          <canvas id="knob_lc_slope" width="110" height="110"></canvas>
          <div class="label">Low Cut Slope</div>
          <div class="value"><span id="val_lc_slope">__DEF_LC_SLOPE__</span> dB/oct</div>
        </div>
      </div>

      <div class="hrow">
        <div class="knob">
          <canvas id="knob_ls_gain" width="110" height="110"></canvas>
          <div class="label">Low Shelf Gain</div>
          <div class="value"><span id="val_ls_gain">__DEF_LS_GAIN__</span> dB</div>
        </div>
        <div class="knob">
          <canvas id="knob_ls_hz" width="110" height="110"></canvas>
          <div class="label">Low Shelf Hz</div>
          <div class="value"><span id="val_ls_hz">__DEF_LS_HZ__</span> Hz</div>
        </div>
        <div class="knob">
          <canvas id="knob_ls_slope" width="110" height="110"></canvas>
          <div class="label">Low Shelf Slope</div>
          <div class="value"><span id="val_ls_slope">__DEF_LS_SLOPE__</span> dB/oct</div>
        </div>
      </div>

      <div class="hrow">
        <div class="knob">
          <canvas id="knob_mid_gain" width="110" height="110"></canvas>
          <div class="label">Mid Gain</div>
          <div class="value"><span id="val_mid_gain">__DEF_MID_GAIN__</span> dB</div>
        </div>
        <div class="knob">
          <canvas id="knob_mid_hz" width="110" height="110"></canvas>
          <div class="label">Mid Hz</div>
          <div class="value"><span id="val_mid_hz">__DEF_MID_HZ__</span> Hz</div>
        </div>
        <div class="knob">
          <canvas id="knob_mid_q" width="110" height="110"></canvas>
          <div class="label">Mid Q</div>
          <div class="value"><span id="val_mid_q">__DEF_MID_Q__</span></div>
        </div>
      </div>

      <div class="hrow">
        <div class="knob">
          <canvas id="knob_hs_gain" width="110" height="110"></canvas>
          <div class="label">High Shelf Gain</div>
          <div class="value"><span id="val_hs_gain">__DEF_HS_GAIN__</span> dB</div>
        </div>
        <div class="knob">
          <canvas id="knob_hs_hz" width="110" height="110"></canvas>
          <div class="label">High Shelf Hz</div>
          <div class="value"><span id="val_hs_hz">__DEF_HS_HZ__</span> Hz</div>
        </div>
        <div class="knob">
          <canvas id="knob_hs_slope" width="110" height="110"></canvas>
          <div class="label">High Shelf Slope</div>
          <div class="value"><span id="val_hs_slope">__DEF_HS_SLOPE__</span> dB/oct</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Hall Reverb -->
  <div class="grid">
    <div class="panel">
      <div class="row">
        <div class="title">Hall Reverb</div>
        <div class="btn" id="btnReverb">Reverb: __DEF_REVERB_ENABLED_TEXT__</div>
      </div>
      <div class="hrow">
        <div class="knob">
          <canvas id="knob_rvb_mix" width="110" height="110"></canvas>
          <div class="label">Mix</div>
          <div class="value"><span id="val_rvb_mix">__DEF_RVB_MIX__</span> %</div>
        </div>
        <div class="knob">
          <canvas id="knob_rvb_t60" width="110" height="110"></canvas>
          <div class="label">Decay</div>
          <div class="value"><span id="val_rvb_t60">__DEF_RVB_T60__</span> s</div>
        </div>
        <div class="knob">
          <canvas id="knob_rvb_damp" width="110" height="110"></canvas>
          <div class="label">Damp</div>
          <div class="value"><span id="val_rvb_damp">__DEF_RVB_DAMP__</span> %</div>
        </div>
        <div class="knob">
          <canvas id="knob_rvb_width" width="110" height="110"></canvas>
          <div class="label">Width</div>
          <div class="value"><span id="val_rvb_width">__DEF_RVB_WIDTH__</span> %</div>
        </div>
      </div>
      <div class="hrow">
        <div class="knob">
          <canvas id="knob_rvb_lc_hz" width="110" height="110"></canvas>
          <div class="label">LoCut Hz (wet)</div>
          <div class="value"><span id="val_rvb_lc_hz">__DEF_RVB_LC_HZ__</span> Hz</div>
        </div>
        <div class="knob">
          <canvas id="knob_rvb_mod_rate" width="110" height="110"></canvas>
          <div class="label">Mod Rate</div>
          <div class="value"><span id="val_rvb_mod_rate">__DEF_RVB_MOD_RATE__</span> Hz</div>
        </div>
        <div class="knob">
          <canvas id="knob_rvb_mod_depth" width="110" height="110"></canvas>
          <div class="label">Mod Depth</div>
          <div class="value"><span id="val_rvb_mod_depth">__DEF_RVB_MOD_DEPTH__</span> ms</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Tape/Mod Delay -->
  <div class="grid">
    <div class="panel">
      <div class="row">
        <div class="title">Tape / Mod Delay</div>
        <div class="btn" id="btnDelay">Delay: __DEF_DELAY_ENABLED_TEXT__</div>
        <div class="btn" id="btnDelayPing">PingPong: __DEF_DLY_PINGPONG_TEXT__</div>
      </div>
      <div class="hrow">
        <div class="knob">
          <canvas id="knob_dly_mix" width="110" height="110"></canvas>
          <div class="label">Mix</div>
          <div class="value"><span id="val_dly_mix">__DEF_DLY_MIX__</span> %</div>
        </div>
        <div class="knob">
          <canvas id="knob_dly_time" width="110" height="110"></canvas>
          <div class="label">Time</div>
          <div class="value"><span id="val_dly_time">__DEF_DLY_TIME_MS__</span> ms</div>
        </div>
        <div class="knob">
          <canvas id="knob_dly_fb" width="110" height="110"></canvas>
          <div class="label">Feedback</div>
          <div class="value"><span id="val_dly_fb">__DEF_DLY_FEEDBACK__</span> %</div>
        </div>
        <div class="knob">
          <canvas id="knob_dly_lowcut" width="110" height="110"></canvas>
          <div class="label">Lowcut</div>
          <div class="value"><span id="val_dly_lowcut">__DEF_DLY_LOWCUT_HZ__</span> Hz</div>
        </div>
        <div class="knob">
          <canvas id="knob_dly_highcut" width="110" height="110"></canvas>
          <div class="label">Highcut</div>
          <div class="value"><span id="val_dly_highcut">__DEF_DLY_HIGHCUT_HZ__</span> Hz</div>
        </div>
        <div class="knob">
          <canvas id="knob_dly_width" width="110" height="110"></canvas>
          <div class="label">Width</div>
          <div class="value"><span id="val_dly_width">__DEF_DLY_WIDTH__</span> %</div>
        </div>
      </div>
      <div class="hrow">
        <div class="knob">
          <canvas id="knob_dly_mod_rate" width="110" height="110"></canvas>
          <div class="label">Mod Rate</div>
          <div class="value"><span id="val_dly_mod_rate">__DEF_DLY_MOD_RATE__</span> Hz</div>
        </div>
        <div class="knob">
          <canvas id="knob_dly_mod_depth" width="110" height="110"></canvas>
          <div class="label">Mod Depth</div>
          <div class="value"><span id="val_dly_mod_depth">__DEF_DLY_MOD_DEPTH__</span> ms</div>
        </div>
        <div class="knob">
          <canvas id="knob_dly_drive" width="110" height="110"></canvas>
          <div class="label">Drive</div>
          <div class="value"><span id="val_dly_drive">__DEF_DLY_DRIVE_DB__</span> dB</div>
        </div>
      </div>
    </div>
  </div>

   <!-- Stereo Chorus -->
  <div class="grid">
    <div class="panel">
      <div class="row">
        <div class="title">Stereo Chorus</div>
        <div class="btn" id="btnChorus">Chorus: __DEF_CHORUS_ENABLED_TEXT__</div>
      </div>
      <div class="hrow">
        <div class="knob">
          <canvas id="knob_ch_mix" width="110" height="110"></canvas>
          <div class="label">Mix</div>
          <div class="value"><span id="val_ch_mix">__DEF_CH_MIX__</span> %</div>
        </div>
        <div class="knob">
          <canvas id="knob_ch_rate" width="110" height="110"></canvas>
          <div class="label">Rate</div>
          <div class="value"><span id="val_ch_rate">__DEF_CH_RATE_HZ__</span> Hz</div>
        </div>
        <div class="knob">
          <canvas id="knob_ch_depth" width="110" height="110"></canvas>
          <div class="label">Depth</div>
          <div class="value"><span id="val_ch_depth">__DEF_CH_DEPTH_MS__</span> ms</div>
        </div>
        <div class="knob">
          <canvas id="knob_ch_delay" width="110" height="110"></canvas>
          <div class="label">Delay</div>
          <div class="value"><span id="val_ch_delay">__DEF_CH_DELAY_MS__</span> ms</div>
        </div>
        <div class="knob">
          <canvas id="knob_ch_width" width="110" height="110"></canvas>
          <div class="label">Width</div>
          <div class="value"><span id="val_ch_width">__DEF_CH_WIDTH__</span> %</div>
        </div>
        <div class="knob">
          <canvas id="knob_ch_fb" width="110" height="110"></canvas>
          <div class="label">Feedback</div>
          <div class="value"><span id="val_ch_fb">__DEF_CH_FEEDBACK__</span> %</div>
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
        <div class="btn" id="btnPickup">Pickup: PIEZO</div>
        <div class="btn" id="btnFxLoop">FX Loop: OFF</div>
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
const MAX_MAIN_KNOB_CHANGE = __MAX_MAIN_KNOB_CHANGE__;
const MAX_PICK_ATTACK_KNOB_CHANGE = __MAX_PICK_ATTACK_KNOB_CHANGE__;
const MAX_PICK_RESO_KNOB_CHANGE = __MAX_PICK_RESO_KNOB_CHANGE__;
const MAX_IR_KNOB_CHANGE = __MAX_IR_KNOB_CHANGE__;

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
    const nv = Math.round((clamp(v, this.min, this.max))/this.step)*this.step;
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
  startDrag(e){ e.preventDefault(); this.dragging = true; this.lastY = e.clientY; this.lastX = e.clientX; this.startVal = this.value; }
  moveDrag(e){
    if (!this.dragging) return;
    const dx = e.clientX - this.lastX;
    const dy = this.lastY - e.clientY;
    this.lastX = e.clientX; this.lastY = e.clientY;
    const useX = Math.abs(dx) > Math.abs(dy);
    const pix = useX ? dx : dy;
    const range = this.max - this.min;
    const delta = pix * this.sens * range / 200.0;
    this.set(this.value + delta, true);
  }
  endDrag(e){ if (!this.dragging) return; this.dragging = false; }
  onWheel(e){ e.preventDefault(); const delta = (e.deltaY < 0 ? this.step : -this.step); this.set(this.value + delta, true); }
  onDoubleClick(e){
    const id = this.c.id;
    const defaults = {
      'knob_main': __DEF_MAIN_DB__,
      'knob_pa': __DEF_PA_DB__, 'knob_pr': __DEF_PR_DB__, 'knob_ir': __DEF_IR_DB__, 'knob_wet': __DEF_WET_PCT__,
      'knob_outg': __DEF_OUTG_DB__,
      'knob_lc_hz': __DEF_LC_HZ__, 'knob_lc_slope': __DEF_LC_SLOPE__,
      'knob_ls_gain': __DEF_LS_GAIN__, 'knob_ls_hz': __DEF_LS_HZ__, 'knob_ls_slope': __DEF_LS_SLOPE__,
      'knob_mid_gain': __DEF_MID_GAIN__, 'knob_mid_hz': __DEF_MID_HZ__, 'knob_mid_q': __DEF_MID_Q__,
      'knob_hs_gain': __DEF_HS_GAIN__, 'knob_hs_hz': __DEF_HS_HZ__, 'knob_hs_slope': __DEF_HS_SLOPE__,

      // Reverb defaults
      'knob_rvb_mix': __DEF_RVB_MIX__, 'knob_rvb_t60': __DEF_RVB_T60__, 'knob_rvb_damp': __DEF_RVB_DAMP__,
      'knob_rvb_width': __DEF_RVB_WIDTH__,
      'knob_rvb_lc_hz': __DEF_RVB_LC_HZ__, 'knob_rvb_mod_rate': __DEF_RVB_MOD_RATE__, 'knob_rvb_mod_depth': __DEF_RVB_MOD_DEPTH__,

      // Delay defaults
      'knob_dly_mix': __DEF_DLY_MIX__, 'knob_dly_time': __DEF_DLY_TIME_MS__, 'knob_dly_fb': __DEF_DLY_FEEDBACK__,
      'knob_dly_lowcut': __DEF_DLY_LOWCUT_HZ__, 'knob_dly_highcut': __DEF_DLY_HIGHCUT_HZ__, 'knob_dly_width': __DEF_DLY_WIDTH__,
      'knob_dly_mod_rate': __DEF_DLY_MOD_RATE__, 'knob_dly_mod_depth': __DEF_DLY_MOD_DEPTH__, 'knob_dly_drive': __DEF_DLY_DRIVE_DB__,

      // Chorus defaults
      'knob_ch_mix': __DEF_CH_MIX__, 'knob_ch_rate': __DEF_CH_RATE_HZ__,
      'knob_ch_depth': __DEF_CH_DEPTH_MS__, 'knob_ch_delay': __DEF_CH_DELAY_MS__,
      'knob_ch_width': __DEF_CH_WIDTH__, 'knob_ch_fb': __DEF_CH_FEEDBACK__,

      // Feedback Protection defaults
      'knob_afb_shift': __DEF_AFB_SHIFT_HZ__,

      // Gate defaults
      'knob_gate_thresh': __DEF_GATE_THRESH_DB__, 'knob_gate_hyst': __DEF_GATE_HYST_DB__,
      'knob_gate_range': __DEF_GATE_RANGE_DB__,
      'knob_gate_attack': __DEF_GATE_ATTACK_MS__, 'knob_gate_release': __DEF_GATE_RELEASE_MS__, 'knob_gate_hold': __DEF_GATE_HOLD_MS__,
      'knob_gate_hpf': __DEF_GATE_HPF_HZ__
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
const $btnPickup = el('btnPickup');
const $btnFxLoop = el('btnFxLoop');
const $btnPhase  = el('btnPhase');
const $btnReverb   = el('btnReverb');
const $btnGate   = el('btnGate');
const $btnDelay = el('btnDelay');
const $btnDelayPing = el('btnDelayPing');
const $btnChorus = el('btnChorus');
const $btnAFB = el('btnAFB');

const kb = {
  main: new Knob(el('knob_main'), {min:-MAX_MAIN_KNOB_CHANGE,max:MAX_MAIN_KNOB_CHANGE,step:1,value:__DEF_MAIN_DB__,prec:0,displayEl:el('val_main'),color:'#ffd166', onchange:(v)=>sendSet({main:v})}),
  pa: new Knob(el('knob_pa'),  {min:-MAX_PICK_ATTACK_KNOB_CHANGE,max:MAX_PICK_ATTACK_KNOB_CHANGE,step:1,value:__DEF_PA_DB__,prec:0,displayEl:el('val_pa'),color:'#ffcc00', onchange:(v)=>sendSet({pa:v})}),
  pr: new Knob(el('knob_pr'),  {min:-MAX_PICK_RESO_KNOB_CHANGE,max:MAX_PICK_RESO_KNOB_CHANGE,step:1,value:__DEF_PR_DB__,prec:0,displayEl:el('val_pr'),color:'#ff7f50', onchange:(v)=>sendSet({pr:v})}),
  ir: new Knob(el('knob_ir'),  {min:-MAX_IR_KNOB_CHANGE,max:MAX_IR_KNOB_CHANGE,step:1,value:__DEF_IR_DB__,prec:0,displayEl:el('val_ir'),color:'#7ee787', onchange:(v)=>sendSet({ir:v})}),
  wet:new Knob(el('knob_wet'), {min:0,max:100,step:1,value:__DEF_WET_PCT__,prec:0,displayEl:el('val_wet'),color:'#00d1ff', onchange:(v)=>sendSet({wet:v})}),

  outg: new Knob(el('knob_outg'), {min:-24,max:24,step:1,value:__DEF_OUTG_DB__,prec:0,displayEl:el('val_outg'),color:'#c586f7', onchange:(v)=>sendSet({outg:v})}),

  lc_hz:    new Knob(el('knob_lc_hz'),    {min:10,max:300,step:1,value:__DEF_LC_HZ__,prec:0,displayEl:el('val_lc_hz'),color:'#e5c07b', onchange:(v)=>sendSet({lc_hz:v})}),
  lc_slope: new Knob(el('knob_lc_slope'), {min:6,max:36,step:6,value:__DEF_LC_SLOPE__,prec:0,displayEl:el('val_lc_slope'),color:'#e5c07b', onchange:(v)=>sendSet({lc_slope:v})}),

  ls_gain:  new Knob(el('knob_ls_gain'),  {min:-18,max:18,step:1,value:__DEF_LS_GAIN__,prec:0,displayEl:el('val_ls_gain'),color:'#98c379', onchange:(v)=>sendSet({ls_gain:v})}),
  ls_hz:    new Knob(el('knob_ls_hz'),    {min:50,max:1000,step:5,value:__DEF_LS_HZ__,prec:0,displayEl:el('val_ls_hz'),color:'#98c379', onchange:(v)=>sendSet({ls_hz:v})}),
  ls_slope: new Knob(el('knob_ls_slope'), {min:6,max:36,step:6,value:__DEF_LS_SLOPE__,prec:0,displayEl:el('val_ls_slope'),color:'#98c379', onchange:(v)=>sendSet({ls_slope:v})}),

  mid_gain: new Knob(el('knob_mid_gain'), {min:-18,max:18,step:1,value:__DEF_MID_GAIN__,prec:0,displayEl:el('val_mid_gain'),color:'#61afef', onchange:(v)=>sendSet({mid_gain:v})}),
  mid_hz:   new Knob(el('knob_mid_hz'),   {min:200,max:6000,step:10,value:__DEF_MID_HZ__,prec:0,displayEl:el('val_mid_hz'),color:'#61afef', onchange:(v)=>sendSet({mid_hz:v})}),
  mid_q:    new Knob(el('knob_mid_q'),    {min:0.5,max:2.0,step:0.05,value:__DEF_MID_Q__,prec:2,displayEl:el('val_mid_q'),color:'#61afef', onchange:(v)=>sendSet({mid_q:v})}),

  hs_gain:  new Knob(el('knob_hs_gain'),  {min:-18,max:18,step:1,value:__DEF_HS_GAIN__,prec:0,displayEl:el('val_hs_gain'),color:'#d19a66', onchange:(v)=>sendSet({hs_gain:v})}),
  hs_hz:    new Knob(el('knob_hs_hz'),    {min:2000,max:16000,step:50,value:__DEF_HS_HZ__,prec:0,displayEl:el('val_hs_hz'),color:'#d19a66', onchange:(v)=>sendSet({hs_hz:v})}),
  hs_slope: new Knob(el('knob_hs_slope'), {min:6,max:36,step:6,value:__DEF_HS_SLOPE__,prec:0,displayEl:el('val_hs_slope'),color:'#d19a66', onchange:(v)=>sendSet({hs_slope:v})}),

  rvb_mix:  new Knob(el('knob_rvb_mix'),  {min:0,max:100,step:1,value:__DEF_RVB_MIX__,prec:0,displayEl:el('val_rvb_mix'),color:'#00b3ff', onchange:(v)=>sendSet({rvb_mix:v})}),
  rvb_t60:  new Knob(el('knob_rvb_t60'),  {min:0.1,max:__DEF_RVB_MAX_DECAY_SEC__,step:0.1,value:__DEF_RVB_T60__,prec:1,displayEl:el('val_rvb_t60'),color:'#00b3ff', onchange:(v)=>sendSet({rvb_t60:v})}),
  rvb_damp: new Knob(el('knob_rvb_damp'), {min:0,max:100,step:1,value:__DEF_RVB_DAMP__,prec:0,displayEl:el('val_rvb_damp'),color:'#00b3ff', onchange:(v)=>sendSet({rvb_damp:v})}),
  rvb_width: new Knob(el('knob_rvb_width'), {min:0,max:100,step:1,value:__DEF_RVB_WIDTH__,prec:0,displayEl:el('val_rvb_width'),color:'#00b3ff', onchange:(v)=>sendSet({rvb_width:v})}),

  rvb_lc_hz: new Knob(el('knob_rvb_lc_hz'), {min:10,max:__DEF_RVB_LC_MAX_HZ__,step:1,value:__DEF_RVB_LC_HZ__,prec:0,displayEl:el('val_rvb_lc_hz'),color:'#00b3ff', onchange:(v)=>sendSet({rvb_lc_hz:v})}),
  rvb_mod_rate: new Knob(el('knob_rvb_mod_rate'), {min:0.05,max:__DEF_RVB_MOD_MAX_RATE_HZ__,step:0.01,value:__DEF_RVB_MOD_RATE__,prec:2,displayEl:el('val_rvb_mod_rate'),color:'#00b3ff', onchange:(v)=>sendSet({rvb_mod_rate:v})}),
  rvb_mod_depth: new Knob(el('knob_rvb_mod_depth'), {min:0.0,max:__DEF_RVB_MOD_MAX_DEPTH_MS__,step:0.1,value:__DEF_RVB_MOD_DEPTH__,prec:1,displayEl:el('val_rvb_mod_depth'),color:'#00b3ff', onchange:(v)=>sendSet({rvb_mod_depth:v})}),

  // Delay knobs
  dly_mix:   new Knob(el('knob_dly_mix'), {min:0,max:100,step:1,value:__DEF_DLY_MIX__,prec:0,displayEl:el('val_dly_mix'),color:'#00ffa0', onchange:(v)=>sendSet({dly_mix:v})}),
  dly_time:  new Knob(el('knob_dly_time'), {min:5,max:__DEF_DLY_TIME_MAX_MS__,step:1,value:__DEF_DLY_TIME_MS__,prec:0,displayEl:el('val_dly_time'),color:'#00ffa0', onchange:(v)=>sendSet({dly_time:v})}),
  dly_fb:    new Knob(el('knob_dly_fb'), {min:0,max:98,step:1,value:__DEF_DLY_FEEDBACK__,prec:0,displayEl:el('val_dly_fb'),color:'#00ffa0', onchange:(v)=>sendSet({dly_feedback:v})}),
  dly_lowcut: new Knob(el('knob_dly_lowcut'), {min:__DEF_DLY_LOWCUT_MIN_HZ__,max:__DEF_DLY_LOWCUT_MAX_HZ__,step:10,value:__DEF_DLY_LOWCUT_HZ__,prec:0,displayEl:el('val_dly_lowcut'),color:'#00ffa0', onchange:(v)=>sendSet({dly_lowcut:v})}),
  dly_highcut: new Knob(el('knob_dly_highcut'), {min:__DEF_DLY_HIGHCUT_MIN_HZ__,max:__DEF_DLY_HIGHCUT_MAX_HZ__,step:10,value:__DEF_DLY_HIGHCUT_HZ__,prec:0,displayEl:el('val_dly_highcut'),color:'#00ffa0', onchange:(v)=>sendSet({dly_highcut:v})}),
  dly_width: new Knob(el('knob_dly_width'), {min:0,max:100,step:1,value:__DEF_DLY_WIDTH__,prec:0,displayEl:el('val_dly_width'),color:'#00ffa0', onchange:(v)=>sendSet({dly_width:v})}),
  dly_mod_rate: new Knob(el('knob_dly_mod_rate'), {min:0.01,max:__DEF_DLY_MOD_MAX_RATE_HZ__,step:0.01,value:__DEF_DLY_MOD_RATE__,prec:2,displayEl:el('val_dly_mod_rate'),color:'#00ffa0', onchange:(v)=>sendSet({dly_mod_rate:v})}),
  dly_mod_depth: new Knob(el('knob_dly_mod_depth'), {min:0.0,max:__DEF_DLY_MOD_MAX_DEPTH_MS__,step:0.1,value:__DEF_DLY_MOD_DEPTH__,prec:1,displayEl:el('val_dly_mod_depth'),color:'#00ffa0', onchange:(v)=>sendSet({dly_mod_depth:v})}),
  dly_drive: new Knob(el('knob_dly_drive'), {min:0,max:18,step:1,value:__DEF_DLY_DRIVE_DB__,prec:0,displayEl:el('val_dly_drive'),color:'#00ffa0', onchange:(v)=>sendSet({dly_drive:v})}),

  // Chorus knobs
  ch_mix:   new Knob(el('knob_ch_mix'), {min:0,max:100,step:1,value:__DEF_CH_MIX__,prec:0,displayEl:el('val_ch_mix'),color:'#37c6ff', onchange:(v)=>sendSet({ch_mix:v})}),
  ch_rate:  new Knob(el('knob_ch_rate'), {min:0.05,max:__DEF_CH_RATE_MAX_HZ__,step:0.01,value:__DEF_CH_RATE_HZ__,prec:2,displayEl:el('val_ch_rate'),color:'#37c6ff', onchange:(v)=>sendSet({ch_rate:v})}),
  ch_depth: new Knob(el('knob_ch_depth'), {min:0.0,max:__DEF_CH_DEPTH_MAX_MS__,step:0.1,value:__DEF_CH_DEPTH_MS__,prec:1,displayEl:el('val_ch_depth'),color:'#37c6ff', onchange:(v)=>sendSet({ch_depth:v})}),
  ch_delay: new Knob(el('knob_ch_delay'), {min:__DEF_CH_DELAY_MIN_MS__,max:__DEF_CH_DELAY_MAX_MS__,step:0.1,value:__DEF_CH_DELAY_MS__,prec:1,displayEl:el('val_ch_delay'),color:'#37c6ff', onchange:(v)=>sendSet({ch_delay:v})}),
  ch_width: new Knob(el('knob_ch_width'), {min:0,max:100,step:1,value:__DEF_CH_WIDTH__,prec:0,displayEl:el('val_ch_width'),color:'#37c6ff', onchange:(v)=>sendSet({ch_width:v})}),
  ch_fb:    new Knob(el('knob_ch_fb'), {min:0,max:__DEF_CH_FEEDBACK_MAX_PCT__,step:1,value:__DEF_CH_FEEDBACK__,prec:0,displayEl:el('val_ch_fb'),color:'#37c6ff', onchange:(v)=>sendSet({ch_feedback:v})}),

  // Feedback Protection Shift knob
  afb_shift: new Knob(el('knob_afb_shift'), {
  min: __DEF_AFB_SHIFT_MIN_HZ__, max: __DEF_AFB_SHIFT_MAX_HZ__, step: 0.1,
  value: __DEF_AFB_SHIFT_HZ__, prec: 1, displayEl: el('val_afb_shift'),
  color: '#ff5577', onchange: (v)=>sendSet({afb_shift:v})
  }),

  // Noise Gate knobs
  gate_thresh: new Knob(el('knob_gate_thresh'), {min:-80,max:-0,step:1,value:__DEF_GATE_THRESH_DB__,prec:0,displayEl:el('val_gate_thresh'),color:'#ffa600', onchange:(v)=>sendSet({gate_thresh:v})}),
  gate_hyst:   new Knob(el('knob_gate_hyst'),   {min:0,max:24,step:0.5,value:__DEF_GATE_HYST_DB__,prec:1,displayEl:el('val_gate_hyst'),color:'#ff9e00', onchange:(v)=>sendSet({gate_hyst:v})}),
  gate_range:  new Knob(el('knob_gate_range'),  {min:6,max:90,step:1,value:__DEF_GATE_RANGE_DB__,prec:0,displayEl:el('val_gate_range'),color:'#ff7b00', onchange:(v)=>sendSet({gate_range:v})}),
  gate_attack: new Knob(el('knob_gate_attack'), {min:0.1,max:200,step:0.1,value:__DEF_GATE_ATTACK_MS__,prec:1,displayEl:el('val_gate_attack'),color:'#ff6d00', onchange:(v)=>sendSet({gate_attack:v})}),
  gate_release:new Knob(el('knob_gate_release'),{min:10,max:1200,step:5,value:__DEF_GATE_RELEASE_MS__,prec:0,displayEl:el('val_gate_release'),color:'#ff6d00', onchange:(v)=>sendSet({gate_release:v})}),
  gate_hold:   new Knob(el('knob_gate_hold'),   {min:0,max:500,step:5,value:__DEF_GATE_HOLD_MS__,prec:0,displayEl:el('val_gate_hold'),color:'#ff6d00', onchange:(v)=>sendSet({gate_hold:v})}),
  gate_hpf:    new Knob(el('knob_gate_hpf'),    {min:20,max:600,step:5,value:__DEF_GATE_HPF_HZ__,prec:0,displayEl:el('val_gate_hpf'),color:'#ff6d00', onchange:(v)=>sendSet({gate_hpf:v})})
};

function sendSet(obj) {
  const params = new URLSearchParams();
  if ('main' in obj) params.set('main', obj.main);
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

  // Reverb
  if ('rvb_enabled' in obj) params.set('rvb_enabled', obj.rvb_enabled ? 1 : 0);
  if ('rvb_mix' in obj) params.set('rvb_mix', obj.rvb_mix);
  if ('rvb_t60' in obj) params.set('rvb_t60', obj.rvb_t60);
  if ('rvb_damp' in obj) params.set('rvb_damp', obj.rvb_damp);
  if ('rvb_width' in obj) params.set('rvb_width', obj.rvb_width);
  if ('rvb_lc_hz' in obj) params.set('rvb_lc_hz', obj.rvb_lc_hz);
  if ('rvb_mod_rate' in obj) params.set('rvb_mod_rate', obj.rvb_mod_rate);
  if ('rvb_mod_depth' in obj) params.set('rvb_mod_depth', obj.rvb_mod_depth);

  // Delay
  if ('dly_enabled' in obj) params.set('dly_enabled', obj.dly_enabled ? 1 : 0);
  if ('dly_pingpong' in obj) params.set('dly_pingpong', obj.dly_pingpong ? 1 : 0);
  if ('dly_mix' in obj) params.set('dly_mix', obj.dly_mix);
  if ('dly_time' in obj) params.set('dly_time', obj.dly_time);
  if ('dly_feedback' in obj) params.set('dly_feedback', obj.dly_feedback);
  if ('dly_lowcut' in obj) params.set('dly_lowcut', obj.dly_lowcut);
  if ('dly_highcut' in obj) params.set('dly_highcut', obj.dly_highcut);
  if ('dly_width' in obj) params.set('dly_width', obj.dly_width);
  if ('dly_mod_rate' in obj) params.set('dly_mod_rate', obj.dly_mod_rate);
  if ('dly_mod_depth' in obj) params.set('dly_mod_depth', obj.dly_mod_depth);
  if ('dly_drive' in obj) params.set('dly_drive', obj.dly_drive);

  // Chorus
  if ('ch_enabled' in obj) params.set('ch_enabled', obj.ch_enabled ? 1 : 0);
  if ('ch_mix' in obj) params.set('ch_mix', obj.ch_mix);
  if ('ch_rate' in obj) params.set('ch_rate', obj.ch_rate);
  if ('ch_depth' in obj) params.set('ch_depth', obj.ch_depth);
  if ('ch_delay' in obj) params.set('ch_delay', obj.ch_delay);
  if ('ch_width' in obj) params.set('ch_width', obj.ch_width);
  if ('ch_feedback' in obj) params.set('ch_feedback', obj.ch_feedback);

  // Feedback Protection
  if ('afb_enabled' in obj) params.set('afb_enabled', obj.afb_enabled ? 1 : 0);
  if ('afb_shift' in obj)   params.set('afb_shift', obj.afb_shift);

  // Noise Gate
  if ('gate_enabled' in obj) params.set('gate_enabled', obj.gate_enabled ? 1 : 0);
  if ('gate_thresh' in obj) params.set('gate_thresh', obj.gate_thresh);
  if ('gate_hyst' in obj) params.set('gate_hyst', obj.gate_hyst);
  if ('gate_range' in obj) params.set('gate_range', obj.gate_range);
  if ('gate_attack' in obj) params.set('gate_attack', obj.gate_attack);
  if ('gate_release' in obj) params.set('gate_release', obj.gate_release);
  if ('gate_hold' in obj) params.set('gate_hold', obj.gate_hold);
  if ('gate_hpf' in obj) params.set('gate_hpf', obj.gate_hpf);

  fetch('/set?' + params.toString()).catch(()=>{});
}

function setMode(m) { fetch('/mode?v='+m).catch(()=>{}); }
function setGuided(on) { fetch('/guided?v='+(on?1:0)).catch(()=>{}); }
function setBypass(on) { fetch('/bypass?v='+(on?1:0)).catch(()=>{}); }
function setPhase(on) { fetch('/set?phase='+(on?1:0)).catch(()=>{}); }
function setPickup(mag) { fetch('/set?pickup='+(mag?1:0)).catch(()=>{}); }
function setFxLoop(on) { fetch('/set?fxloop='+(on?1:0)).catch(()=>{}); }

$btnPickup.addEventListener('click', ()=>{
  const isMag = $btnPickup.classList.contains('on'); setPickup(!isMag);
});
$btnFxLoop.addEventListener('click', ()=>{
  const isOn = $btnFxLoop.classList.contains('on'); setFxLoop(!isOn);
});

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

$btnAFB.addEventListener('click', ()=>{
  const isOn = $btnAFB.classList.contains('on'); sendSet({afb_enabled: !isOn});
});

$btnGate.addEventListener('click', ()=>{
  const isOn = $btnGate.classList.contains('on'); sendSet({gate_enabled: !isOn});
});
$btnReverb.addEventListener('click', ()=>{
  const isOn = $btnReverb.classList.contains('on'); sendSet({rvb_enabled: !isOn});
});
$btnDelay.addEventListener('click', ()=>{
  const isOn = $btnDelay.classList.contains('on'); sendSet({dly_enabled: !isOn});
});
$btnDelayPing.addEventListener('click', ()=>{
  const isOn = $btnDelayPing.classList.contains('on'); sendSet({dly_pingpong: !isOn});
});

$btnChorus.addEventListener('click', ()=>{
  const isOn = $btnChorus.classList.contains('on'); sendSet({ch_enabled: !isOn});
});

function applyStatus(s){
  if (typeof s !== 'object') return;
  kb.main.set(s.main_dB, false);
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

  // Reverb
  if ('rvb_enabled' in s) {
    const revOn = !!s.rvb_enabled;
    $btnReverb.textContent = 'Reverb: ' + (revOn ? 'ON' : 'OFF');
    $btnReverb.classList.toggle('on', revOn);
    $btnReverb.classList.toggle('off', !revOn);
  }
  if ('rvb_mix_pct' in s) kb.rvb_mix.set(s.rvb_mix_pct, false);
  if ('rvb_t60_sec' in s) kb.rvb_t60.set(s.rvb_t60_sec, false);
  if ('rvb_damp_pct' in s) kb.rvb_damp.set(s.rvb_damp_pct, false);
  if ('rvb_width_pct' in s) kb.rvb_width.set(s.rvb_width_pct, false);

  if ('rvb_lc_hz' in s) kb.rvb_lc_hz.set(s.rvb_lc_hz, false);
  if ('rvb_mod_rate_hz' in s) kb.rvb_mod_rate.set(s.rvb_mod_rate_hz, false);
  if ('rvb_mod_depth_ms' in s) kb.rvb_mod_depth.set(s.rvb_mod_depth_ms, false);

  // Delay
  if ('dly_enabled' in s) {
    const don = !!s.dly_enabled;
    $btnDelay.textContent = 'Delay: ' + (don ? 'ON' : 'OFF');
    $btnDelay.classList.toggle('on', don);
    $btnDelay.classList.toggle('off', !don);
  }
  if ('dly_pingpong' in s) {
    const pong = !!s.dly_pingpong;
    $btnDelayPing.textContent = 'PingPong: ' + (pong ? 'ON' : 'OFF');
    $btnDelayPing.classList.toggle('on', pong);
    $btnDelayPing.classList.toggle('off', !pong);
  }
  if ('dly_mix_pct' in s) kb.dly_mix.set(s.dly_mix_pct, false);
  if ('dly_time_ms' in s) kb.dly_time.set(s.dly_time_ms, false);
  if ('dly_feedback_pct' in s) kb.dly_fb.set(s.dly_feedback_pct, false);
  if ('dly_lowcut_hz' in s) kb.dly_lowcut.set(s.dly_lowcut_hz, false);
  if ('dly_highcut_hz' in s) kb.dly_highcut.set(s.dly_highcut_hz, false);
  if ('dly_width_pct' in s) kb.dly_width.set(s.dly_width_pct, false);
  if ('dly_mod_rate_hz' in s) kb.dly_mod_rate.set(s.dly_mod_rate_hz, false);
  if ('dly_mod_depth_ms' in s) kb.dly_mod_depth.set(s.dly_mod_depth_ms, false);
  if ('dly_drive_db' in s) kb.dly_drive.set(s.dly_drive_db, false);

  // Chorus
  if ('ch_enabled' in s) {
    const con = !!s.ch_enabled;
    $btnChorus.textContent = 'Chorus: ' + (con ? 'ON' : 'OFF');
    $btnChorus.classList.toggle('on', con);
    $btnChorus.classList.toggle('off', !con);
  }
  if ('ch_mix_pct' in s) kb.ch_mix.set(s.ch_mix_pct, false);
  if ('ch_rate_hz' in s) kb.ch_rate.set(s.ch_rate_hz, false);
  if ('ch_depth_ms' in s) kb.ch_depth.set(s.ch_depth_ms, false);
  if ('ch_delay_ms' in s) kb.ch_delay.set(s.ch_delay_ms, false);
  if ('ch_width_pct' in s) kb.ch_width.set(s.ch_width_pct, false);
  if ('ch_feedback_pct' in s) kb.ch_fb.set(s.ch_feedback_pct, false);

  // Phase switch
  const phaseOn = (s.phaseInvert01 >= 0.5);
  $btnPhase.textContent = 'Phase Invert: ' + (phaseOn ? 'ON' : 'OFF');
  $btnPhase.classList.toggle('on', phaseOn);
  $btnPhase.classList.toggle('off', !phaseOn);

  // Feedback Protection
  if ('afb_enabled' in s) {
    const aon = !!s.afb_enabled;
    $btnAFB.textContent = 'Feedback Prot: ' + (aon ? 'ON' : 'OFF');
    $btnAFB.classList.toggle('on', aon);
    $btnAFB.classList.toggle('off', !aon);
  }
  if ('afb_shift_hz' in s) kb.afb_shift.set(s.afb_shift_hz, false);

  // Gate
  if ('gate_enabled' in s) {
    const gon = !!s.gate_enabled;
    $btnGate.textContent = 'Gate: ' + (gon ? 'ON' : 'OFF');
    $btnGate.classList.toggle('on', gon);
    $btnGate.classList.toggle('off', !gon);
  }
  if ('gate_thresh_db' in s) kb.gate_thresh.set(s.gate_thresh_db, false);
  if ('gate_hyst_db' in s) kb.gate_hyst.set(s.gate_hyst_db, false);
  if ('gate_range_db' in s) kb.gate_range.set(s.gate_range_db, false);
  if ('gate_attack_ms' in s) kb.gate_attack.set(s.gate_attack_ms, false);
  if ('gate_release_ms' in s) kb.gate_release.set(s.gate_release_ms, false);
  if ('gate_hold_ms' in s) kb.gate_hold.set(s.gate_hold_ms, false);
  if ('gate_hpf_hz' in s) kb.gate_hpf.set(s.gate_hpf_hz, false);

  // Seg
  $modeSeg.querySelectorAll('.segbtn').forEach(b=>{
    if (parseInt(b.dataset.mode,10) === s.mode) b.classList.add('active');
    else b.classList.remove('active');
  });

  el('modeName').textContent = s.modeName;
  el('guidedState').textContent = s.guidedAuto ? 'ON' : 'OFF';
  el('bypassState').textContent = s.bypass ? 'ON' : 'OFF';

  if ('pickup01' in s) {
    const mag = (s.pickup01 >= 0.5);
    $btnPickup.textContent = 'Pickup: ' + (mag ? 'MAGNETIC' : 'PIEZO');
    $btnPickup.classList.toggle('on', mag);
    $btnPickup.classList.toggle('off', !mag);
  }
  if ('fxLoop' in s) {
    $btnFxLoop.textContent = 'FX Loop: ' + (s.fxLoop ? 'ON' : 'OFF');
    $btnFxLoop.classList.toggle('on', !!s.fxLoop);
    $btnFxLoop.classList.toggle('off', !s.fxLoop);
  }

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
