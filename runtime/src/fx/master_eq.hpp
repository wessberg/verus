#pragma once
// Master EQ: post-model IIR biquads (mirrors src/verus/eq.py).

#include "common.hpp"
#include "fx/reverb.hpp"

// ───────────────────────── MASTER EQ (post-model IIR) ─────────────────────────
// Mirrors master_eq_sos() in src/verus/eq.py exactly (same RBJ formulas, same slope mapping), so
// `--process --eq ...` and the pedal produce the same curve. The EQ used to be four in-graph FIR
// banks; 33–193-tap FIRs cannot realise 20–300 Hz cuts and the banks were ~35% of the model's
// MACs, so it now lives here.
//   Low cut : bypass at <= 20 Hz; slope 6 = 1st-order HP, 12/24/36 = Butterworth biquad cascade
//   Shelves / mid bell: RBJ cookbook; 0 dB = bypass
struct Biquad {
  float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f, a1 = 0.0f, a2 = 0.0f;
  float z1 = 0.0f, z2 = 0.0f;   // transposed direct form II state
  inline float process(float x) {
    const float y = b0 * x + z1;
    z1 = b1 * x - a1 * y + z2;
    z2 = b2 * x - a2 * y;
    return y;
  }
  void reset() { z1 = 0.0f; z2 = 0.0f; }
};

static inline void rbjFirstOrderHP(Biquad& q, float sr, float f0) {
  const double w0 = 2.0 * M_PI * double(clampf(f0, 1.0f, 0.49f * sr)) / double(sr);
  const double k = std::tan(w0 / 2.0);
  const double n = 1.0 / (1.0 + k);
  q.b0 = float(n); q.b1 = float(-n); q.b2 = 0.0f; q.a1 = float((k - 1.0) * n); q.a2 = 0.0f;
}

enum class RbjKind { HP2, LowShelf, HighShelf, Peak };

static inline void rbjBiquad(Biquad& q, RbjKind kind, float sr, float f0, float Q, float gain_db) {
  const double w0 = 2.0 * M_PI * double(clampf(f0, 1.0f, 0.49f * sr)) / double(sr);
  const double cw = std::cos(w0), sw = std::sin(w0);
  const double alpha = sw / (2.0 * std::max(double(Q), 1e-3));
  const double A = std::pow(10.0, double(gain_db) / 40.0);
  double b0, b1, b2, a0, a1, a2;
  switch (kind) {
    case RbjKind::HP2:
      b0 = (1 + cw) / 2; b1 = -(1 + cw); b2 = (1 + cw) / 2;
      a0 = 1 + alpha;    a1 = -2 * cw;   a2 = 1 - alpha;
      break;
    case RbjKind::Peak:
      b0 = 1 + alpha * A; b1 = -2 * cw; b2 = 1 - alpha * A;
      a0 = 1 + alpha / A; a1 = -2 * cw; a2 = 1 - alpha / A;
      break;
    case RbjKind::LowShelf: {
      const double sa = 2.0 * std::sqrt(A) * alpha;
      b0 = A * ((A + 1) - (A - 1) * cw + sa); b1 =  2 * A * ((A - 1) - (A + 1) * cw); b2 = A * ((A + 1) - (A - 1) * cw - sa);
      a0 =      (A + 1) + (A - 1) * cw + sa;  a1 = -2 *     ((A - 1) + (A + 1) * cw); a2 =      (A + 1) + (A - 1) * cw - sa;
      break; }
    case RbjKind::HighShelf:
    default: {
      const double sa = 2.0 * std::sqrt(A) * alpha;
      b0 = A * ((A + 1) + (A - 1) * cw + sa); b1 = -2 * A * ((A - 1) + (A + 1) * cw); b2 = A * ((A + 1) + (A - 1) * cw - sa);
      a0 =      (A + 1) - (A - 1) * cw + sa;  a1 =  2 *     ((A - 1) - (A + 1) * cw); a2 =      (A + 1) - (A - 1) * cw - sa;
      break; }
  }
  q.b0 = float(b0 / a0); q.b1 = float(b1 / a0); q.b2 = float(b2 / a0); q.a1 = float(a1 / a0); q.a2 = float(a2 / a0);
}

// RBJ shelf "slope" S in (0,1]: 12 dB/oct ~ S=1 (steepest without overshoot)
static inline float rbjShelfQFromSlope(float gain_db, float slope_dbpoct) {
  const double A = std::pow(10.0, double(gain_db) / 40.0);
  const double S = std::min(1.0, std::max(0.25, double(slope_dbpoct) / 12.0));
  return float(1.0 / std::sqrt((A + 1.0 / A) * (1.0 / S - 1.0) + 2.0));
}

struct MasterEqIIR {
  // Fixed slots by role so a knob change never moves a section's state into another role:
  //   0..2 low-cut sections, 3 low shelf, 4 mid bell, 5 high shelf
  static constexpr int kSlots = 6;
  static constexpr float kLcBypassHz = 20.0f;
  Biquad sec[kSlots];
  bool   active[kSlots] = {false, false, false, false, false, false};
  bool   anyActive = false;
  float  sr = kSR;
  // cached knob values (coefficients are recomputed only on change)
  float c_lc_hz = -1.0f, c_lc_slope = -1.0f, c_ls_g = -999.0f, c_ls_hz = -1.0f, c_ls_slope = -1.0f,
        c_mid_g = -999.0f, c_mid_hz = -1.0f, c_mid_q = -1.0f, c_hs_g = -999.0f, c_hs_hz = -1.0f, c_hs_slope = -1.0f;

  void reset() { for (auto& s : sec) s.reset(); }

  static float nearestSlope(float s) {
    const float cands[4] = {6.0f, 12.0f, 24.0f, 36.0f};
    float best = cands[0];
    for (float c : cands) if (std::fabs(c - s) < std::fabs(best - s)) best = c;
    return best;
  }

  // Rebuild sections from gKnobs (called per block from the audio callback; cheap unless changed)
  void updateFromKnobs() {
    const float lc_hz = gKnobs.lc_hz.load(std::memory_order_relaxed);
    const float lc_sl = gKnobs.lc_slope_dbpoct.load(std::memory_order_relaxed);
    const float ls_g  = gKnobs.ls_gain_db.load(std::memory_order_relaxed);
    const float ls_hz = gKnobs.ls_hz.load(std::memory_order_relaxed);
    const float ls_sl = gKnobs.ls_slope_dbpoct.load(std::memory_order_relaxed);
    const float md_g  = gKnobs.mid_gain_db.load(std::memory_order_relaxed);
    const float md_hz = gKnobs.mid_hz.load(std::memory_order_relaxed);
    const float md_q  = gKnobs.mid_q.load(std::memory_order_relaxed);
    const float hs_g  = gKnobs.hs_gain_db.load(std::memory_order_relaxed);
    const float hs_hz = gKnobs.hs_hz.load(std::memory_order_relaxed);
    const float hs_sl = gKnobs.hs_slope_dbpoct.load(std::memory_order_relaxed);
    const bool same = lc_hz == c_lc_hz && lc_sl == c_lc_slope && ls_g == c_ls_g && ls_hz == c_ls_hz && ls_sl == c_ls_slope &&
                      md_g == c_mid_g && md_hz == c_mid_hz && md_q == c_mid_q && hs_g == c_hs_g && hs_hz == c_hs_hz && hs_sl == c_hs_slope;
    if (same) return;
    c_lc_hz = lc_hz; c_lc_slope = lc_sl; c_ls_g = ls_g; c_ls_hz = ls_hz; c_ls_slope = ls_sl;
    c_mid_g = md_g; c_mid_hz = md_hz; c_mid_q = md_q; c_hs_g = hs_g; c_hs_hz = hs_hz; c_hs_slope = hs_sl;

    bool act[kSlots] = {false, false, false, false, false, false};
    if (lc_hz > kLcBypassHz + 1e-6f) {
      const float slope = nearestSlope(lc_sl);
      if (slope == 6.0f) {
        rbjFirstOrderHP(sec[0], sr, lc_hz); act[0] = true;
      } else {
        static const float q12[1] = {0.70710678f};
        static const float q24[2] = {0.54119610f, 1.30656296f};
        static const float q36[3] = {0.51763809f, 0.70710678f, 1.93185165f};
        const float* qs = (slope == 12.0f) ? q12 : (slope == 24.0f) ? q24 : q36;
        const int nq = (slope == 12.0f) ? 1 : (slope == 24.0f) ? 2 : 3;
        for (int i = 0; i < nq; ++i) { rbjBiquad(sec[i], RbjKind::HP2, sr, lc_hz, qs[i], 0.0f); act[i] = true; }
      }
    }
    if (std::fabs(ls_g) > 1e-6f) { rbjBiquad(sec[3], RbjKind::LowShelf,  sr, ls_hz, rbjShelfQFromSlope(ls_g, ls_sl), ls_g); act[3] = true; }
    if (std::fabs(md_g) > 1e-6f) { rbjBiquad(sec[4], RbjKind::Peak,      sr, md_hz, md_q, md_g);                        act[4] = true; }
    if (std::fabs(hs_g) > 1e-6f) { rbjBiquad(sec[5], RbjKind::HighShelf, sr, hs_hz, rbjShelfQFromSlope(hs_g, hs_sl), hs_g); act[5] = true; }
    anyActive = false;
    for (int i = 0; i < kSlots; ++i) {
      if (!act[i]) sec[i].reset();       // a section that just switched off starts clean next time
      active[i] = act[i];
      anyActive = anyActive || act[i];
    }
  }

  inline void processBlock(float* x, size_t n) {
    if (!anyActive) return;
    for (size_t i = 0; i < n; ++i) {
      float v = x[i];
      for (int s = 0; s < kSlots; ++s) if (active[s]) v = sec[s].process(v);
      x[i] = std::isfinite(v) ? v : 0.0f;
    }
  }
};
static MasterEqIIR gMasterEq;
