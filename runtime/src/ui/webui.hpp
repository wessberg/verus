#pragma once
// Knob web UI: a tiny HTTP server bound to 127.0.0.1.

#include "common.hpp"
#include "fx/anti_feedback.hpp"
#include "fx/chorus.hpp"
#include "ui/webui_html.hpp"

// ───────────────────────── KNOB WEB UI (SEPARATE WINDOW VIA BROWSER) ─────────────────────────
// Enabled with `--webui [port]` (default 8088, bound to 127.0.0.1 only). Runs in its own
// non-RT thread; it only reads/writes the same atomics as the keyboard handler.

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

static std::string ui_jsonStatus() {
  std::ostringstream oss;
  oss << "{";
  // stems
  oss << "\"main_dB\":"      << gKnobs.main_dB.load()        << ",";
  oss << "\"pickAttack_dB\":"<< gKnobs.pickAttack_dB.load()  << ",";
  oss << "\"pickReso_dB\":"  << gKnobs.pickReso_dB.load()    << ",";
  oss << "\"ir_dB\":"        << gKnobs.ir_dB.load()          << ",";
  oss << "\"wet_pct\":"      << gKnobs.wet_pct.load()        << ",";

  // Output / phase
  oss << "\"outGain_dB\":"   << gKnobs.outGain_dB.load()     << ",";
  oss << "\"phaseInvert01\":"<< gKnobs.phaseInvert01.load()  << ",";

  // Master EQ
  oss << "\"lc_hz\":"        << gKnobs.lc_hz.load()          << ",";
  oss << "\"lc_slope_dbpoct\":"<< gKnobs.lc_slope_dbpoct.load() << ",";

  oss << "\"ls_gain_db\":"   << gKnobs.ls_gain_db.load()     << ",";
  oss << "\"ls_hz\":"        << gKnobs.ls_hz.load()          << ",";
  oss << "\"ls_slope_dbpoct\":"<< gKnobs.ls_slope_dbpoct.load() << ",";

  oss << "\"mid_gain_db\":"  << gKnobs.mid_gain_db.load()    << ",";
  oss << "\"mid_hz\":"       << gKnobs.mid_hz.load()         << ",";
  oss << "\"mid_q\":"        << gKnobs.mid_q.load()          << ",";

  oss << "\"hs_gain_db\":"   << gKnobs.hs_gain_db.load()     << ",";
  oss << "\"hs_hz\":"        << gKnobs.hs_hz.load()          << ",";
  oss << "\"hs_slope_dbpoct\":"<< gKnobs.hs_slope_dbpoct.load() << ",";

  // Reverb
  {
    bool rvb_enabled=true; float rvb_mix=0, rvb_t60=0, rvb_damp=0, rvb_width=0, rvb_lc_hz=0, rvb_mod_rate=0, rvb_mod_depth_ms=0;
    ReverbUIGet(rvb_enabled, rvb_mix, rvb_t60, rvb_damp, rvb_width, rvb_lc_hz, rvb_mod_rate, rvb_mod_depth_ms);
    oss << "\"rvb_enabled\":"   << (rvb_enabled?1:0) << ",";
    oss << "\"rvb_mix_pct\":"  << rvb_mix   << ",";
    oss << "\"rvb_t60_sec\":"  << rvb_t60   << ",";
    oss << "\"rvb_damp_pct\":" << rvb_damp  << ",";
    oss << "\"rvb_width_pct\":"<< rvb_width << ",";
    oss << "\"rvb_lc_hz\":"        << rvb_lc_hz       << ",";
    oss << "\"rvb_mod_rate_hz\":"  << rvb_mod_rate   << ",";
    oss << "\"rvb_mod_depth_ms\":" << rvb_mod_depth_ms << ",";
  }

    // Delay
  {
    bool dly_enabled=false, dly_ping=false;
    float mix=0, time_ms=0, fb=0, hc=0, width=0, rate=0, depth=0, drive=0, lc=0;
    DelayUIGet(dly_enabled, mix, time_ms, fb, hc, width, dly_ping, rate, depth, drive, lc);
    oss << "\"dly_enabled\":"     << (dly_enabled ? 1 : 0) << ",";
    oss << "\"dly_pingpong\":"    << (dly_ping ? 1 : 0) << ",";
    oss << "\"dly_mix_pct\":"     << mix << ",";
    oss << "\"dly_time_ms\":"     << time_ms << ",";
    oss << "\"dly_feedback_pct\":"<< fb << ",";
    oss << "\"dly_lowcut_hz\":"   << lc << ",";
    oss << "\"dly_highcut_hz\":"  << hc << ",";
    oss << "\"dly_width_pct\":"   << width << ",";
    oss << "\"dly_mod_rate_hz\":" << rate << ",";
    oss << "\"dly_mod_depth_ms\":"<< depth << ",";
    oss << "\"dly_drive_db\":"    << drive << ",";
  }

    // Chorus
  {
    bool ch_enabled=false; float ch_mix=0, ch_rate=0, ch_depth=0, ch_delay=0, ch_width=0, ch_fb=0;
    ChorusUIGet(ch_enabled, ch_mix, ch_rate, ch_depth, ch_delay, ch_width, ch_fb);
    oss << "\"ch_enabled\":"      << (ch_enabled ? 1 : 0) << ",";
    oss << "\"ch_mix_pct\":"      << ch_mix   << ",";
    oss << "\"ch_rate_hz\":"      << ch_rate  << ",";
    oss << "\"ch_depth_ms\":"     << ch_depth << ",";
    oss << "\"ch_delay_ms\":"     << ch_delay << ",";
    oss << "\"ch_width_pct\":"    << ch_width << ",";
    oss << "\"ch_feedback_pct\":" << ch_fb    << ",";
  }

  // Feedback Protection
  {
    oss << "\"afb_enabled\":" << (gKnobs.afb_enabled.load() ? 1 : 0) << ",";
    oss << "\"afb_shift_hz\":" << gKnobs.afb_shift_hz.load() << ",";
  }

  // Noise Gate
  {
    bool gate_enabled=false; float gth=0, gh=0, ga=0, gr=0, ghld=0, grng=0, ghpf=0;
    GateUIGet(gate_enabled, gth, gh, ga, gr, ghld, grng, ghpf);
    oss << "\"gate_enabled\":"        << (gate_enabled ? 1 : 0) << ",";
    oss << "\"gate_thresh_db\":" << gth   << ",";
    oss << "\"gate_hyst_db\":"   << gh    << ",";
    oss << "\"gate_attack_ms\":" << ga    << ",";
    oss << "\"gate_release_ms\":"<< gr    << ",";
    oss << "\"gate_hold_ms\":"   << ghld  << ",";
    oss << "\"gate_range_db\":"  << grng  << ",";
    oss << "\"gate_hpf_hz\":"    << ghpf  << ",";
  }

  // switches/mode
  oss << "\"bypass\":"       << (gBypass.load()?1:0) << ",";
  oss << "\"mode\":"         << gStyleMode.load()    << ",";
  oss << "\"modeName\":"     << ui_styleModeName_jsonsafe() << ",";
  oss << "\"pickup01\":"     << gKnobs.pickup01.load() << ",";
  oss << "\"fxLoop\":"       << (gFxLoopEnabled.load()?1:0) << ",";
  oss << "\"guidedAuto\":"   << (gGuidedAuto.load()?1:0);
  oss << "}";
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

  float mainv = gKnobs.main_dB.load();
  getf("main", mainv, -MAX_MAIN_KNOB_CHANGE, MAX_MAIN_KNOB_CHANGE);
  gKnobs.main_dB.store(mainv);

  float pa = gKnobs.pickAttack_dB.load();
  float pr = gKnobs.pickReso_dB.load();
  float ir = gKnobs.ir_dB.load();
  float wt = gKnobs.wet_pct.load();
  getf("pa", pa, -MAX_PICK_ATTACK_KNOB_CHANGE, MAX_PICK_ATTACK_KNOB_CHANGE);
  getf("pr", pr, -MAX_PICK_RESO_KNOB_CHANGE, MAX_PICK_RESO_KNOB_CHANGE);
  getf("ir", ir, -MAX_IR_KNOB_CHANGE, MAX_IR_KNOB_CHANGE);
  getf("wet", wt, 0.0f, 100.0f);
  gKnobs.pickAttack_dB.store(pa);
  gKnobs.pickReso_dB.store(pr);
  gKnobs.ir_dB.store(ir);
  gKnobs.wet_pct.store(wt);

  // Output/phase
  float outg = gKnobs.outGain_dB.load();
  getf("outg", outg, -24.0f, 24.0f);
  gKnobs.outGain_dB.store(outg);

  auto itPh = q.find("phase");
  if (itPh != q.end()) {
    try {
      int v = std::stoi(itPh->second);
      gKnobs.phaseInvert01.store((v!=0) ? 1 : 0);
    } catch (...) {}
  }

  // Pickup type (0 = piezo, 1 = magnetic) and FX loop switch
  auto itPu = q.find("pickup");
  if (itPu != q.end()) {
    try { gKnobs.pickup01.store(std::stoi(itPu->second) != 0 ? 1.0f : 0.0f); } catch (...) {}
  }
  auto itFx = q.find("fxloop");
  if (itFx != q.end()) {
    try { gFxLoopEnabled.store(std::stoi(itFx->second) != 0, std::memory_order_relaxed); } catch (...) {}
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

  // Reverb (partial updates allowed)
  {
    bool rvb_enabled = true;
    float rvb_mix = 0.0f, rvb_t60 = 0.0f, rvb_damp = 0.0f, rvb_width = 0.0f, rvb_lc_hz = 0.0f, rvb_mod_rate = 0.0f, rvb_mod_depth = 0.0f;
    ReverbUIGet(rvb_enabled, rvb_mix, rvb_t60, rvb_damp, rvb_width, rvb_lc_hz, rvb_mod_rate, rvb_mod_depth);

    // Toggle enable
    auto itRvbEnabled = q.find("rvb_enabled");
    if (itRvbEnabled != q.end()) {
      try {
        int v = std::stoi(itRvbEnabled->second);
        int prev = RvbEnabledStore().load(std::memory_order_relaxed);
        if (v != prev) {
          RvbEnabledStore().store(v ? 1 : 0, std::memory_order_relaxed);
          ReverbReset(); // avoid pops on change
        }
        rvb_enabled = (v != 0);
      } catch (...) {}
    }
    getf("rvb_mix",  rvb_mix, 0.0f, 100.0f);
    getf("rvb_t60",  rvb_t60, 0.10f, rvb_max_decay_sec);
    getf("rvb_damp", rvb_damp, 0.0f, 100.0f);
    getf("rvb_width", rvb_width, 0.0f, 100.0f);
    getf("rvb_lc_hz", rvb_lc_hz, 10.0f, rvb_lc_max_hz);
    getf("rvb_mod_rate", rvb_mod_rate, 0.05f, rvb_mod_max_rate_hz);
    getf("rvb_mod_depth", rvb_mod_depth, 0.0f, rvb_mod_max_depth_ms);
    ReverbUISet(rvb_enabled, rvb_mix, rvb_t60, rvb_damp, rvb_width, rvb_lc_hz, rvb_mod_rate, rvb_mod_depth);
  }

    // Delay (partial updates allowed)
  {
    bool dly_enabled=false, dly_ping=false;
    float mix=0, time_ms=0, fb=0, hc=0, lc=0, width=0, rate=0, depth=0, drive=0;
    DelayUIGet(dly_enabled, mix, time_ms, fb, hc, width, dly_ping, rate, depth, drive, lc);

    // Toggle delay enable; bump reset on change
    auto itDlyEnabled = q.find("dly_enabled");
    if (itDlyEnabled != q.end()) {
      try {
        int v = std::stoi(itDlyEnabled->second);
        int prev = DlyEnabledStore().load(std::memory_order_relaxed);
        if (v != prev) {
          DlyEnabledStore().store(v ? 1 : 0, std::memory_order_relaxed);
          DelayReset();
        }
        dly_enabled = (v != 0);
      } catch (...) {}
    }

    // Toggle ping-pong
    auto itDlyPing = q.find("dly_pingpong");
    if (itDlyPing != q.end()) {
      try {
        int v = std::stoi(itDlyPing->second);
        DlyPingPongStore().store(v ? 1 : 0, std::memory_order_relaxed);
        dly_ping = (v != 0);
      } catch (...) {}
    }

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

    getf("dly_mix",       mix, 0.0f, 100.0f);
    getf("dly_time",      time_ms, 5.0f, dly_time_max_ms);
    getf("dly_feedback",  fb,   0.0f, dly_max_feedback_pct);
    getf("dly_lowcut",    lc,   dly_lowcut_min_hz, dly_lowcut_max_hz);
    getf("dly_highcut",   hc,   dly_highcut_min_hz, dly_highcut_max_hz);
    getf("dly_width",     width,0.0f, 100.0f);
    getf("dly_mod_rate",  rate, 0.01f, dly_mod_max_rate_hz);
    getf("dly_mod_depth", depth,0.0f, dly_mod_max_depth_ms);
    getf("dly_drive",     drive,0.0f, 18.0f);

    DelayUISet(dly_enabled, mix, time_ms, fb, hc, lc, width, dly_ping, rate, depth, drive);
  }

    // Chorus (partial updates allowed)
  {
    bool ch_enabled=false;
    float mix=0, rate=0, depth=0, delay=0, width=0, fb=0;
    ChorusUIGet(ch_enabled, mix, rate, depth, delay, width, fb);

    auto itChEnabled = q.find("ch_enabled");
    if (itChEnabled != q.end()) {
      try {
        int v = std::stoi(itChEnabled->second);
        int prev = ChEnabledStore().load(std::memory_order_relaxed);
        if (v != prev) {
          ChEnabledStore().store(v ? 1 : 0, std::memory_order_relaxed);
          ChorusReset();
        }
        ch_enabled = (v != 0);
      } catch (...) {}
    }

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
    getf("ch_mix",      mix,   0.0f, 100.0f);
    getf("ch_rate",     rate,  0.05f, ch_rate_max_hz);
    getf("ch_depth",    depth, 0.0f,  ch_depth_max_ms);
    getf("ch_delay",    delay, ch_delay_min_ms, ch_delay_max_ms);
    getf("ch_width",    width, 0.0f, 100.0f);
    getf("ch_feedback", fb,    0.0f, ch_feedback_max_pct);

    ChorusUISet(ch_enabled, mix, rate, depth, delay, width, fb);
  }

  // Feedback Protection (frequency shifter)
  {
    auto itAFBEnabled = q.find("afb_enabled");
    if (itAFBEnabled != q.end()) {
      try {
        int v = std::stoi(itAFBEnabled->second);
        int prev = AFBEnabledStore().load(std::memory_order_relaxed);
        if (v != prev) {
          AFBEnabledStore().store(v ? 1 : 0, std::memory_order_relaxed);
          AFBReset();
        }
      } catch (...) {}
    }
    float afb_shift = AFBShiftHzStore().load(std::memory_order_relaxed);
    auto itAFBShift = q.find("afb_shift");
    if (itAFBShift != q.end()) {
      try {
        float v = std::stof(itAFBShift->second);
        if (v < afb_shift_min_hz) v = afb_shift_min_hz;
        if (v > afb_shift_max_hz) v = afb_shift_max_hz;
        afb_shift = v;
      } catch (...) {}
    }
    AFBShiftHzStore().store(afb_shift, std::memory_order_relaxed);
  }

  // Noise Gate
  {
    bool gon=false; float gth=0, gh=0, ga=0, gr=0, ghld=0, grng=0, ghpf=0;
    GateUIGet(gon, gth, gh, ga, gr, ghld, grng, ghpf);

    // Parse gate_enabled (toggle) first and bump reset on change
    auto itGateEnabled = q.find("gate_enabled");
    if (itGateEnabled != q.end()) {
      try {
        int v = std::stoi(itGateEnabled->second);
        int prev = GateEnabledStore().load(std::memory_order_relaxed);
        if (v != prev) {
          GateEnabledStore().store(v ? 1 : 0, std::memory_order_relaxed);
          GateReset(); // avoid pops on state change
        }
        gon = (v != 0);
      } catch (...) {}
    }

    getf("gate_thresh", gth, -80.0f, -0.0f);
    getf("gate_hyst",   gh,   0.0f,   24.0f);
    getf("gate_attack", ga,   0.10f,  200.0f);
    getf("gate_release",gr,  10.0f,  1200.0f);
    getf("gate_hold",   ghld, 0.0f,   500.0f);
    getf("gate_range",  grng, 6.0f,   90.0f);
    getf("gate_hpf",    ghpf, 20.0f,  600.0f);

    GateUISet(gon, gth, gh, ga, gr, ghld, grng, ghpf);
  }
}

static inline void ui_handleMode(const std::unordered_map<std::string,std::string>& q) {
  auto it = q.find("v");
  if (it != q.end()) {
    try {
      int v = std::stoi(it->second);
      gStyleMode.store(std::max(0, std::min(2, v)));
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

static inline std::string fmtInt(float v) {
  int iv = static_cast<int>(std::lround(v));
  return std::to_string(iv);
}
static inline std::string fmtFloatTrim(float v, int prec) {
  std::ostringstream os;
  os.setf(std::ios::fixed);
  os << std::setprecision(prec) << v;
  std::string s = os.str();
  if (prec > 0) {
    size_t dot = s.find('.');
    if (dot != std::string::npos) {
      while (!s.empty() && s.back() == '0') s.pop_back();
      if (!s.empty() && s.back() == '.') s.pop_back();
    }
  }
  return s;
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
    size_t pos = req.find("\r\n\r\n");
    if (pos != std::string::npos) { headerEnd = int(pos + 4); break; }
    if (req.size() > 16384) break;
  }
  if (headerEnd < 0) {
    std::string h = ui_httpHeader("400 Bad Request", "text/plain; charset=utf-8", 0);
    ui_writeAllStr(cfd, h);
    return;
  }
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

    // Limits
    ui_replaceAll(body, "__MAX_MAIN_KNOB_CHANGE__", std::to_string(MAX_MAIN_KNOB_CHANGE));
    ui_replaceAll(body, "__MAX_PICK_ATTACK_KNOB_CHANGE__", std::to_string(MAX_PICK_ATTACK_KNOB_CHANGE));
    ui_replaceAll(body, "__MAX_PICK_RESO_KNOB_CHANGE__", std::to_string(MAX_PICK_RESO_KNOB_CHANGE));
    ui_replaceAll(body, "__MAX_IR_KNOB_CHANGE__", std::to_string(MAX_IR_KNOB_CHANGE));

    // Current defaults from back-end
    const float def_main = gKnobs.main_dB.load();
    const float def_pa   = gKnobs.pickAttack_dB.load();
    const float def_pr   = gKnobs.pickReso_dB.load();
    const float def_ir   = gKnobs.ir_dB.load();
    const float def_wet  = gKnobs.wet_pct.load();
    const float def_outg = gKnobs.outGain_dB.load();
    const float def_phase= gKnobs.phaseInvert01.load();

    const float def_lc_hz   = gKnobs.lc_hz.load();
    const float def_lc_slope= gKnobs.lc_slope_dbpoct.load();

    const float def_ls_gain = gKnobs.ls_gain_db.load();
    const float def_ls_hz   = gKnobs.ls_hz.load();
    const float def_ls_slope= gKnobs.ls_slope_dbpoct.load();

    const float def_mid_gain= gKnobs.mid_gain_db.load();
    const float def_mid_hz  = gKnobs.mid_hz.load();
    const float def_mid_q   = gKnobs.mid_q.load();

    const float def_hs_gain = gKnobs.hs_gain_db.load();
    const float def_hs_hz   = gKnobs.hs_hz.load();
    const float def_hs_slope= gKnobs.hs_slope_dbpoct.load();

    bool rvb_enabled=true;
    float rvb_mix=0, rvb_t60=0, rvb_damp=0, rvb_width=0, rvb_lc_hz=0, rvb_mod_rate=0, rvb_mod_depth=0;
    ReverbUIGet(rvb_enabled, rvb_mix, rvb_t60, rvb_damp, rvb_width, rvb_lc_hz, rvb_mod_rate, rvb_mod_depth);

    // Delay defaults
    bool dly_enabled=false, dly_ping=false;
    float dly_mix=0, dly_time=0, dly_fb=0, dly_hc=0, dly_width=0, dly_rate=0, dly_depth=0, dly_drive=0, dly_lc=0;
    DelayUIGet(dly_enabled, dly_mix, dly_time, dly_fb, dly_hc, dly_width, dly_ping, dly_rate, dly_depth, dly_drive, dly_lc);

    // Chorus defaults
    bool ch_enabled=false; float ch_mix=0, ch_rate=0, ch_depth=0, ch_delay=0, ch_width=0, ch_fb=0;
    ChorusUIGet(ch_enabled, ch_mix, ch_rate, ch_depth, ch_delay, ch_width, ch_fb);

    // Feedback Protection defaults
    const bool  def_afb_enabled = (gKnobs.afb_enabled.load() != 0);
    const float def_afb_shift   = gKnobs.afb_shift_hz.load();

    // Noise gate defaults
    bool gate_enabled=false; float gth=0, gh=0, ga=0, gr=0, ghld=0, grng=0, ghpf=0;
    GateUIGet(gate_enabled, gth, gh, ga, gr, ghld, grng, ghpf);

    // Inject formatted defaults (integers where appropriate)
    ui_replaceAll(body, "__DEF_MAIN_DB__",   fmtInt(def_main));
    ui_replaceAll(body, "__DEF_PA_DB__",     fmtInt(def_pa));
    ui_replaceAll(body, "__DEF_PR_DB__",     fmtInt(def_pr));
    ui_replaceAll(body, "__DEF_IR_DB__",     fmtInt(def_ir));
    ui_replaceAll(body, "__DEF_WET_PCT__",   fmtInt(def_wet));

    ui_replaceAll(body, "__DEF_OUTG_DB__",   fmtInt(def_outg));
    ui_replaceAll(body, "__DEF_PHASE_TEXT__", (def_phase >= 0.5f) ? "ON" : "OFF");

    ui_replaceAll(body, "__DEF_LC_HZ__",     fmtInt(def_lc_hz));
    ui_replaceAll(body, "__DEF_LC_SLOPE__",  fmtInt(def_lc_slope));

    ui_replaceAll(body, "__DEF_LS_GAIN__",   fmtInt(def_ls_gain));
    ui_replaceAll(body, "__DEF_LS_HZ__",     fmtInt(def_ls_hz));
    ui_replaceAll(body, "__DEF_LS_SLOPE__",  fmtInt(def_ls_slope));

    ui_replaceAll(body, "__DEF_MID_GAIN__",  fmtInt(def_mid_gain));
    ui_replaceAll(body, "__DEF_MID_HZ__",    fmtInt(def_mid_hz));
    ui_replaceAll(body, "__DEF_MID_Q__",     fmtFloatTrim(def_mid_q, 2));

    ui_replaceAll(body, "__DEF_HS_GAIN__",   fmtInt(def_hs_gain));
    ui_replaceAll(body, "__DEF_HS_HZ__",     fmtInt(def_hs_hz));
    ui_replaceAll(body, "__DEF_HS_SLOPE__",  fmtInt(def_hs_slope));

    ui_replaceAll(body, "__DEF_RVB_ENABLED_TEXT__", rvb_enabled ? "ON" : "OFF");
    ui_replaceAll(body, "__DEF_RVB_MIX__",   fmtInt(rvb_mix));
    ui_replaceAll(body, "__DEF_RVB_T60__",   fmtFloatTrim(rvb_t60, 1));
    ui_replaceAll(body, "__DEF_RVB_DAMP__",  fmtInt(rvb_damp));
    ui_replaceAll(body, "__DEF_RVB_WIDTH__", fmtInt(rvb_width));
    ui_replaceAll(body, "__DEF_RVB_MAX_DECAY_SEC__", fmtFloatTrim(rvb_max_decay_sec,1));

    ui_replaceAll(body, "__DEF_RVB_LC_HZ__",   fmtInt(rvb_lc_hz));
    ui_replaceAll(body, "__DEF_RVB_LC_MAX_HZ__",fmtInt(rvb_lc_max_hz));
    ui_replaceAll(body, "__DEF_RVB_MOD_MAX_RATE_HZ__",fmtFloatTrim(rvb_mod_max_rate_hz,2));
    ui_replaceAll(body, "__DEF_RVB_MOD_MAX_DEPTH_MS__",fmtFloatTrim(rvb_mod_max_depth_ms,1));
    ui_replaceAll(body, "__DEF_RVB_MOD_RATE__",fmtFloatTrim(rvb_mod_rate,2));
    ui_replaceAll(body, "__DEF_RVB_MOD_DEPTH__",fmtFloatTrim(rvb_mod_depth,1));

    // Delay defaults + maxima
    ui_replaceAll(body, "__DEF_DELAY_ENABLED_TEXT__", dly_enabled ? "ON" : "OFF");
    ui_replaceAll(body, "__DEF_DLY_PINGPONG_TEXT__", dly_ping ? "ON" : "OFF");
    ui_replaceAll(body, "__DEF_DLY_MIX__", fmtInt(dly_mix));
    ui_replaceAll(body, "__DEF_DLY_TIME_MS__", fmtFloatTrim(dly_time, 0));
    ui_replaceAll(body, "__DEF_DLY_FEEDBACK__", fmtInt(dly_fb));
    ui_replaceAll(body, "__DEF_DLY_HIGHCUT_HZ__", fmtInt(dly_hc));
    ui_replaceAll(body, "__DEF_DLY_WIDTH__", fmtInt(dly_width));
    ui_replaceAll(body, "__DEF_DLY_MOD_RATE__", fmtFloatTrim(dly_rate, 2));
    ui_replaceAll(body, "__DEF_DLY_MOD_DEPTH__", fmtFloatTrim(dly_depth, 1));
    ui_replaceAll(body, "__DEF_DLY_DRIVE_DB__", fmtInt(dly_drive));
    ui_replaceAll(body, "__DEF_DLY_TIME_MAX_MS__", fmtFloatTrim(dly_time_max_ms, 1));
    ui_replaceAll(body, "__DEF_DLY_MOD_MAX_RATE_HZ__", fmtFloatTrim(dly_mod_max_rate_hz, 2));
    ui_replaceAll(body, "__DEF_DLY_MOD_MAX_DEPTH_MS__", fmtFloatTrim(dly_mod_max_depth_ms, 1));
    ui_replaceAll(body, "__DEF_DLY_HIGHCUT_MIN_HZ__", fmtInt(dly_highcut_min_hz));
    ui_replaceAll(body, "__DEF_DLY_HIGHCUT_MAX_HZ__", fmtInt(dly_highcut_max_hz));
    ui_replaceAll(body, "__DEF_DLY_LOWCUT_HZ__", fmtInt(dly_lc));
    ui_replaceAll(body, "__DEF_DLY_LOWCUT_MIN_HZ__", fmtInt(dly_lowcut_min_hz));
    ui_replaceAll(body, "__DEF_DLY_LOWCUT_MAX_HZ__", fmtInt(dly_lowcut_max_hz));

    // Chorus defaults
    ui_replaceAll(body, "__DEF_CHORUS_ENABLED_TEXT__", ch_enabled ? "ON" : "OFF");
    ui_replaceAll(body, "__DEF_CH_MIX__",              fmtInt(ch_mix));
    ui_replaceAll(body, "__DEF_CH_RATE_HZ__",          fmtFloatTrim(ch_rate, 2));
    ui_replaceAll(body, "__DEF_CH_DEPTH_MS__",         fmtFloatTrim(ch_depth, 1));
    ui_replaceAll(body, "__DEF_CH_DELAY_MS__",         fmtFloatTrim(ch_delay, 1));
    ui_replaceAll(body, "__DEF_CH_WIDTH__",            fmtInt(ch_width));
    ui_replaceAll(body, "__DEF_CH_FEEDBACK__",         fmtInt(ch_fb));

    ui_replaceAll(body, "__DEF_CH_RATE_MAX_HZ__",      fmtFloatTrim(ch_rate_max_hz, 2));
    ui_replaceAll(body, "__DEF_CH_DEPTH_MAX_MS__",     fmtFloatTrim(ch_depth_max_ms, 1));
    ui_replaceAll(body, "__DEF_CH_DELAY_MIN_MS__",     fmtFloatTrim(ch_delay_min_ms, 1));
    ui_replaceAll(body, "__DEF_CH_DELAY_MAX_MS__",     fmtFloatTrim(ch_delay_max_ms, 1));
    ui_replaceAll(body, "__DEF_CH_FEEDBACK_MAX_PCT__", fmtInt(ch_feedback_max_pct));

    // Feedback Protection defaults
    ui_replaceAll(body, "__DEF_AFB_ENABLED_TEXT__", def_afb_enabled ? "ON" : "OFF");
    ui_replaceAll(body, "__DEF_AFB_SHIFT_HZ__",     fmtFloatTrim(def_afb_shift, 1));
    ui_replaceAll(body, "__DEF_AFB_SHIFT_MIN_HZ__", fmtFloatTrim(afb_shift_min_hz, 1));
    ui_replaceAll(body, "__DEF_AFB_SHIFT_MAX_HZ__", fmtFloatTrim(afb_shift_max_hz, 1));

    // Noise Gate defaults
    ui_replaceAll(body, "__DEF_GATE_ENABLED_TEXT__", gate_enabled ? "ON" : "OFF");
    ui_replaceAll(body, "__DEF_GATE_THRESH_DB__", fmtInt(gth));
    ui_replaceAll(body, "__DEF_GATE_HYST_DB__",   fmtFloatTrim(gh,1));
    ui_replaceAll(body, "__DEF_GATE_RANGE_DB__",  fmtInt(grng));
    ui_replaceAll(body, "__DEF_GATE_ATTACK_MS__", fmtFloatTrim(ga,1));
    ui_replaceAll(body, "__DEF_GATE_RELEASE_MS__",fmtFloatTrim(gr,0));
    ui_replaceAll(body, "__DEF_GATE_HOLD_MS__",   fmtFloatTrim(ghld,0));
    ui_replaceAll(body, "__DEF_GATE_HPF_HZ__",    fmtInt(ghpf));

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
  // The accept loop polls gWebUIShouldStop / gStopFlag every 200 ms (select timeout)
  gWebUIShouldStop.store(true);
  if (gWebUIThread.joinable()) gWebUIThread.join();
}
