#pragma once
// Noise gate.

#include "common.hpp"
#include "fx/reverb.hpp"

// ───────────────────────── NOISE GATE: UI stores + DSP ─────────────────────────

// Gate UI atomics (kept separate from gKnobs to avoid struct changes)
static std::atomic<int>& GateEnabledStore()            { static std::atomic<int> v{gKnobs.ng_enabled.load()}; return v; } // ON by default
static std::atomic<float>& GateThreshDBStore()    { static std::atomic<float> v{gKnobs.ng_threshold_db.load()}; return v; } // dBFS
static std::atomic<float>& GateHystDBStore()      { static std::atomic<float> v{gKnobs.ng_hysteresis_db.load()}; return v; }   // dB
static std::atomic<float>& GateAttackMsStore()    { static std::atomic<float> v{gKnobs.ng_attack_ms.load()}; return v; }   // ms
static std::atomic<float>& GateReleaseMsStore()   { static std::atomic<float> v{gKnobs.ng_release_ms.load()}; return v; } // ms
static std::atomic<float>& GateHoldMsStore()      { static std::atomic<float> v{gKnobs.ng_hold_ms.load()}; return v; }  // ms
static std::atomic<float>& GateRangeDBStore()     { static std::atomic<float> v{gKnobs.ng_range_db.load()}; return v; }  // dB attenuation when closed
static std::atomic<float>& GateScHPHzStore()      { static std::atomic<float> v{gKnobs.ng_sc_hpf_hz.load()}; return v; } // Hz

// Gate reset epoch (function-backed to avoid new globals)
static std::atomic<uint64_t>& GateResetEpochRef() {
  static std::atomic<uint64_t> e{0};
  return e;
}
static inline void GateReset() {
  GateResetEpochRef().fetch_add(1, std::memory_order_acq_rel);
}
static inline uint64_t GateGetResetEpoch() {
  return GateResetEpochRef().load(std::memory_order_acquire);
}

// Clamp helpers for Gate UI
static inline float clampf_range(float x, float lo, float hi) { return std::max(lo, std::min(hi, x)); }

// UI set/get for gate (thread-safe atomics)
static void GateUISet(bool on, float thresh_db, float hyst_db, float atk_ms, float rel_ms, float hold_ms, float range_db, float sc_hpf_hz) {
  // Hard limits (sane ranges for acoustic guitar)
  int on_i = on ? 1 : 0;
  float th = clampf_range(thresh_db, -80.0f, -0.0f);
  float hy = clampf_range(hyst_db,    0.0f,   24.0f);
  float am = clampf_range(atk_ms,     0.10f,  200.0f);
  float rm = clampf_range(rel_ms,     10.0f,  1200.0f);
  float hm = clampf_range(hold_ms,    0.0f,   500.0f);
  float rg = clampf_range(range_db,    6.0f,   90.0f);
  float hp = clampf_range(sc_hpf_hz,  20.0f,  600.0f);

  GateEnabledStore().store(on_i, std::memory_order_relaxed);
  GateThreshDBStore().store(th, std::memory_order_relaxed);
  GateHystDBStore().store(hy, std::memory_order_relaxed);
  GateAttackMsStore().store(am, std::memory_order_relaxed);
  GateReleaseMsStore().store(rm, std::memory_order_relaxed);
  GateHoldMsStore().store(hm, std::memory_order_relaxed);
  GateRangeDBStore().store(rg, std::memory_order_relaxed);
  GateScHPHzStore().store(hp, std::memory_order_relaxed);
}

static void GateUIGet(bool &on, float &thresh_db, float &hyst_db, float &atk_ms, float &rel_ms, float &hold_ms, float &range_db, float &sc_hpf_hz) {
  on        = (GateEnabledStore().load(std::memory_order_relaxed) != 0);
  thresh_db = GateThreshDBStore().load(std::memory_order_relaxed);
  hyst_db   = GateHystDBStore().load(std::memory_order_relaxed);
  atk_ms    = GateAttackMsStore().load(std::memory_order_relaxed);
  rel_ms    = GateReleaseMsStore().load(std::memory_order_relaxed);
  hold_ms   = GateHoldMsStore().load(std::memory_order_relaxed);
  range_db  = GateRangeDBStore().load(std::memory_order_relaxed);
  sc_hpf_hz = GateScHPHzStore().load(std::memory_order_relaxed);
}

// ───────── NoiseGate DSP ─────────
// Simple, robust gate with peak-ish envelope, attack/release smoothing, hold and hysteresis.
// Sidechain uses a 2x 1st-order HP cascade to ignore very low thumps/rumble.

struct NoiseGate {
  // Detection/envelope
  float env = 0.0f;
  float envAtkA = 0.0f;  // per-sample smoothing coefficients
  float envRelA = 0.0f;
  // Gain smoothing
  float curGain = 1.0f;
  float gAtkA = 0.0f;
  float gRelA = 0.0f;

  // Thresholds and closed gain (linear amplitude)
  float thrOpenLin  = 0.0f;
  float thrCloseLin = 0.0f;
  float closedGainLin = 0.001f; // -60 dB default

  // Hold and state
  int   holdSamples = 0;
  int   holdCounter = 0;
  bool  isOpen = true;

  // Sidechain HPF
  OnePoleHP hp1;
  OnePoleHP hp2;
  float prevHpfHz = -1.0f;

  // Sample rate (use global kSR=48k default)
  float sr = kSR;

  static inline float coeffFromMs(float ms, float sr) {
    float t = std::max(0.0001f, ms * 0.001f);
    float a = 1.0f - std::exp(-1.0f / (t * std::max(1.0f, sr)));
    if (a < 0.000001f) a = 0.000001f;
    if (a > 1.0f) a = 1.0f;
    return a;
  }

  void setSampleRate(float sampleRate) {
    sr = std::max(1000.0f, sampleRate);
  }

  void reset() {
    env = 0.0f;
    curGain = 1.0f;
    holdCounter = 0;
    isOpen = true;
    hp1.reset();
    hp2.reset();
  }

  void setParams(float thresh_db, float hyst_db, float atk_ms, float rel_ms, float hold_ms, float range_db, float sc_hpf_hz) {
    // thresholds in linear amplitude
    float thOpen = db2lin(thresh_db);
    float thClose = db2lin(thresh_db - std::fabs(hyst_db));
    // ensure close <= open (amplitude)
    if (thClose > thOpen) thClose = thOpen;

    thrOpenLin  = thOpen;
    thrCloseLin = thClose;
    closedGainLin = db2lin(-std::fabs(range_db));

    envAtkA = coeffFromMs(atk_ms, sr);
    envRelA = coeffFromMs(rel_ms, sr);
    // Use same time constants for gain smoothing; could be different if desired
    gAtkA   = envAtkA;
    gRelA   = envRelA;

    holdSamples = (int)std::round(std::max(0.0f, hold_ms) * 0.001f * sr);

    float hp = std::max(20.0f, sc_hpf_hz);
    if (std::fabs(hp - prevHpfHz) > 1e-3f) {
      hp1.setCutoff(hp, sr);
      hp2.setCutoff(hp, sr);
      prevHpfHz = hp;
    }
  }

  inline void processBlock(float* x, size_t n, bool enabled) {
    if (!enabled) return;
    for (size_t i = 0; i < n; ++i) {
      float xi = x[i];

      // Sidechain HPF (12 dB/oct via cascade)
      float sc = hp2.process(hp1.process(xi));
      float r  = std::fabs(sc);

      // Envelope follower with separate attack/release
      if (r > env) {
        env += envAtkA * (r - env);
      } else {
        env += envRelA * (r - env);
      }

      // Hysteresis + hold
      if (isOpen) {
        if (env < thrCloseLin) {
          if (holdCounter <= 0) {
            // start closing
            isOpen = false;
          } else {
            holdCounter--;
          }
        } else {
          // signal above close threshold -> refresh hold
          holdCounter = holdSamples;
        }
      } else {
        if (env >= thrOpenLin) {
          isOpen = true;
          holdCounter = holdSamples;
        }
      }

      float target = isOpen ? 1.0f : closedGainLin;
      if (target > curGain) {
        // opening -> attack toward 1.0
        curGain += gAtkA * (target - curGain);
      } else {
        // closing -> release toward closed gain
        curGain += gRelA * (target - curGain);
      }

      x[i] = xi * curGain;
    }
  }
};
