#pragma once
// Stereo hall reverb (Freeverb-style, 48 kHz).

#include "common.hpp"
#include "knobs.hpp"
#include "state.hpp"
#include "util.hpp"

// ───────────────────────── STEREO HALL REVERB (Freeverb-style, 48 kHz tuned) ─────────────────────────
// Mono in -> stereo out (interleaved).
// Enhancements:
//   - Reverb-only low-cut (HPF) pre-tank (mono) so you can take bass out of the reverb without thinning the dry.
//   - Gentle stereo modulation (post-tank, wet path only) for a prettier tail, with adjustable rate/depth.
//   - Proper reset on bypass edges via gReverbResetEpoch.
//
// UI parameters:
//   rvb_mix   (0..1)    wet mix
//   rvb_decay  (0.05..12) decay time to -60 dB
//   rvb_damp  (0..1)    high-frequency damping inside comb feedback
//   rvb_width  (0..1)    stereo width mixer
//   rvb_lc_hz        10..400 Hz     reverb-only low cut (pre-tank)
//   rvb_mod_rate     0.05..3.0 Hz   modulation rate (post-tank, wet only)
//   rvb_mod_depth    0..10 ms       modulation depth (post-tank, wet only)

static constexpr int kNumCombs = 8;
static constexpr int kNumAP    = 4;

static constexpr float rvb_lc_max_hz    = 8000.0f;
static constexpr float rvb_mod_max_rate_hz    = 8.0f;
static constexpr float rvb_mod_max_depth_ms    = 8.0f;
static constexpr float rvb_max_decay_sec = 60.0f;

// Constants
static constexpr float kFixedGain   = 0.02f; // canonical Freeverb input gain
static constexpr float kAPFeedback  = 0.5f;  // allpass feedback
static constexpr float kSR          = 48000.0f;      // fixed 48 kHz

// Reverb UI atomics
static std::atomic<int>& RvbEnabledStore()   { static std::atomic<int> v{gKnobs.rvb_enabled.load()}; return v; } // ON by default
static std::atomic<float>& RvbMixStore()     { static std::atomic<float> v{gKnobs.rvb_mix.load()};   return v; }     // percent
static std::atomic<float>& RvbT60Store()     { static std::atomic<float> v{gKnobs.rvb_decay.load()};  return v; }     // seconds
static std::atomic<float>& RvbDampStore()    { static std::atomic<float> v{gKnobs.rvb_damp.load()};  return v; }     // 0..1
static std::atomic<float>& RvbWidthStore()   { static std::atomic<float> v{gKnobs.rvb_width.load()};   return v; }     // 0..1
static std::atomic<float>& RvbLCHzStore()        { static std::atomic<float> v{gKnobs.rvb_lc_hz.load()}; return v; }  // Hz
static std::atomic<float>& RvbModRateStore()     { static std::atomic<float> v{gKnobs.rvb_mod_rate.load()};  return v; }  // Hz
static std::atomic<float>& RvbModDepthMsStore()  { static std::atomic<float> v{gKnobs.rvb_mod_depth.load()};   return v; }  // ms

static void ReverbUISet(bool enabled, float mix_pct, float t60_sec, float damp_pct, float width_pct, float lc_hz, float mod_rate, float mod_depth_ms) {
  int enabled_int = enabled ? 1 : 0;
  float mix = clampf(mix_pct, 0.0f, 100.0f);
  float t60 = clampf(t60_sec, 0.10f, rvb_max_decay_sec);
  float d01 = clampf(damp_pct * 0.01f, 0.0f, 1.0f);
  float w01 = clampf(width_pct * 0.01f, 0.0f, 1.0f);
  float lc = clampf(lc_hz, 10.0f, rvb_lc_max_hz);
  float mr = clampf(mod_rate, 0.05f, rvb_mod_max_rate_hz);
  float md = clampf(mod_depth_ms, 0.0f, rvb_mod_max_depth_ms);

  RvbEnabledStore().store(enabled_int, std::memory_order_relaxed);
  RvbMixStore().store(mix,   std::memory_order_relaxed);
  RvbT60Store().store(t60,   std::memory_order_relaxed);
  RvbDampStore().store(d01,  std::memory_order_relaxed);
  RvbWidthStore().store(w01, std::memory_order_relaxed);
  RvbLCHzStore().store(lc,   std::memory_order_relaxed);
  RvbModRateStore().store(mr,   std::memory_order_relaxed);
  RvbModDepthMsStore().store(md, std::memory_order_relaxed);
}

static void ReverbUIGet(bool &enabled, float &mix_pct, float &t60_sec, float &damp_pct, float &width_pct, float &lc_hz, float &mod_rate, float &mod_depth_ms) {
  enabled   = (RvbEnabledStore().load(std::memory_order_relaxed) != 0);
  mix_pct   = RvbMixStore().load(std::memory_order_relaxed);
  t60_sec   = RvbT60Store().load(std::memory_order_relaxed);
  damp_pct  = RvbDampStore().load(std::memory_order_relaxed) * 100.0f;
  width_pct = RvbWidthStore().load(std::memory_order_relaxed) * 100.0f;
  lc_hz    = RvbLCHzStore().load(std::memory_order_relaxed);
  mod_rate  = RvbModRateStore().load(std::memory_order_relaxed);
  mod_depth_ms = RvbModDepthMsStore().load(std::memory_order_relaxed);
}

// Internal helper: first-order HP used in cascade for pre-tank low-cut
struct OnePoleHP {
  float a = 0.0f; // alpha = rc / (rc + dt)
  float x1 = 0.0f;
  float y1 = 0.0f;

  void setCutoff(float fc, float sr) {
    float f = std::max(1.0f, fc);
    float dt = 1.0f / std::max(1.0f, sr);
    float rc = 1.0f / (2.0f * float(M_PI) * f);
    a = rc / (rc + dt);
  }
  inline float process(float x) {
    float y = a * (y1 + x - x1);
    x1 = x;
    y1 = y;
    return y;
  }
  void reset() { x1 = 0.0f; y1 = 0.0f; }
};

// Internal helper: lightweight modulated delay for wet tail sweetening
struct ModDelay {
  std::vector<float> buf;
  int w = 0;
  int mask = 0;

  float phase = 0.0f;
  float phaseInc = 0.0f;

  float baseSamp = 0.0f;
  float depthSamp = 0.0f;

  bool enabled = false;

  void init(int minCapacitySamples) {
    int cap = 1;
    while (cap < minCapacitySamples) cap <<= 1;
    buf.assign(cap, 0.0f);
    mask = cap - 1;
    w = 0;
    phase = 0.0f;
  }

  void setParams(float rateHz, float depthMs, float sr, float baseMs, float phaseOffsetRad) {
    const float srSafe = std::max(1.0f, sr);
    float r = std::max(0.0f, rateHz);
    float dms = std::max(0.0f, depthMs);

    baseSamp  = std::max(1.0f, baseMs  * srSafe * 0.001f);
    depthSamp = std::max(0.0f, dms     * srSafe * 0.001f);
    phaseInc  = 2.0f * float(M_PI) * r / srSafe;

    // Keep base/depth inside the ring so we never read the write slot.
    const float maxD = float(mask - 2);
    if (baseSamp < 2.0f) baseSamp = 2.0f;
    if (baseSamp > maxD) baseSamp = maxD;

    float maxDepthDown = baseSamp - 1.0f;
    float maxDepthUp   = maxD - baseSamp;
    float maxDepth     = std::max(0.0f, std::min(maxDepthDown, maxDepthUp));
    if (depthSamp > maxDepth) depthSamp = maxDepth;

    // Keep stereo decorrelation but avoid accumulating offset on each call.
    // Apply the offset once relative to current phase and wrap.
    phase += phaseOffsetRad;
    while (phase >= 2.0f * float(M_PI)) phase -= 2.0f * float(M_PI);
    while (phase < 0.0f) phase += 2.0f * float(M_PI);

    enabled = (depthSamp > 1e-6f && r >= 0.01f);
  }

  inline float readLinear(float D) const {
    // Bound D away from extremes so (r1) never hits the write index
    const float maxD = float(mask - 2);
    if (D < 1.0f) D = 1.0f;
    if (D > maxD) D = maxD;

    int dInt = int(std::floor(D));
    float frac = D - float(dInt);

    // r0 = younger tap = x[n - dInt]
    int r0 = (w - dInt) & mask;
    // r1 = older tap   = x[n - dInt - 1]
    int r1 = (r0 - 1) & mask;

    // Proper linear interpolation: y = r0*(1-frac) + r1*frac
    return buf[r0] * (1.0f - frac) + buf[r1] * frac;
  }

  inline float tick(float x) {
    buf[w] = x;

    float D = baseSamp;
    if (enabled) {
      float m = std::sin(phase);
      phase += phaseInc;
      if (phase >= 2.0f * float(M_PI)) phase -= 2.0f * float(M_PI);
      D += depthSamp * m;
    }

    float y = readLinear(D);
    w = (w + 1) & mask;
    return y;
  }

  void reset() {
    if (!buf.empty()) std::fill(buf.begin(), buf.end(), 0.0f);
    w = 0; phase = 0.0f;
  }
};

static void ReverbProcessMonoToStereo(const float* inMono,
                                      float* outLRInterleaved,
                                      size_t n,
                                      unsigned long /*framesPerBuffer*/,
                                      const PaStreamCallbackTimeInfo* /*timeInfo*/,
                                      bool prewarmOnly = false)
{
  struct Freeverb48Stereo {

    // 48 kHz tuned delays (samples) (scaled and rounded from 44.1k Jezar tables)
    int combDelaysL[kNumCombs]  = {1216,1294,1390,1476,1547,1623,1694,1762};
    int combDelaysR[kNumCombs]  = {1241,1317,1415,1502,1572,1648,1721,1785};
    int allpassDelaysL[kNumAP]  = {605,480,371,245};
    int allpassDelaysR[kNumAP]  = {631,505,396,270};

    struct Comb {
      std::vector<float> buf;
      int idx = 0;
      float filterStore = 0.0f;
      float feedback = 0.0f;
      float damp1 = 0.0f; // (1 - damp)
      float damp2 = 0.0f; // damp
    };
    struct Allpass {
      std::vector<float> buf;
      int idx = 0;
    };

    Comb comb[2][kNumCombs];
    Allpass ap[2][kNumAP];
    bool inited = false;

    // Param cache
    float prevT60 = -1.0f;
    float prevDmp = -1.0f;
    float prevWidth = -1.0f;

    // Reverb-only low cut (cascade of two 1st-order HPF)
    OnePoleHP hp1, hp2;
    float prevLcHz = -1.0f;

    // Modulated wet delay (post-tank)
    ModDelay modL, modR;
    float prevModRate = -1.0f;
    float prevModDepth = -1.0f;
    bool modInited = false;

    // Reset epoch
    uint64_t lastResetSeen = 0;

    void init() {
      // Allocate buffers to exact 48 kHz tuned lengths
      for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < kNumCombs; ++i) {
          const int M = (ch==0) ? combDelaysL[i] : combDelaysR[i];
          comb[ch][i].buf.assign(std::max(32, M), 0.0f);
          comb[ch][i].idx = 0;
          comb[ch][i].filterStore = 0.0f;
        }
        for (int i = 0; i < kNumAP; ++i) {
          const int M = (ch==0) ? allpassDelaysL[i] : allpassDelaysR[i];
          ap[ch][i].buf.assign(std::max(16, M), 0.0f);
          ap[ch][i].idx = 0;
        }
      }
      updateDamping(gKnobs.rvb_damp.load());
      updateFeedbackForT60(gKnobs.rvb_decay.load());
      prevT60   = gKnobs.rvb_decay.load();
      prevDmp   =  gKnobs.rvb_damp.load();
      prevWidth = gKnobs.rvb_width.load();

      // Pre-tank HPF defaults
      float lc = clampf(RvbLCHzStore().load(std::memory_order_relaxed), 10.0f, rvb_lc_max_hz);
      hp1.setCutoff(lc, kSR);
      hp2.setCutoff(lc, kSR);
      hp1.reset(); hp2.reset();
      prevLcHz = lc;

      // Modulated wet sweetener: buffers big enough for ~30 ms
      if (!modInited) {
        modL.init(8192);
        modR.init(8192);
        // default params
        float rate = clampf(RvbModRateStore().load(std::memory_order_relaxed), 0.05f, rvb_mod_max_rate_hz);
        float depth = clampf(RvbModDepthMsStore().load(std::memory_order_relaxed), 0.0f, rvb_mod_max_depth_ms);
        // Slightly different base delay and phase for stereo decorrelation
        modL.setParams(rate, depth, kSR, 12.0f, 0.0f);
        modR.setParams(rate, depth, kSR, 13.7f, float(M_PI) * 0.5f); // +90 deg
        prevModRate  = rate;
        prevModDepth = depth;
        modInited = true;
      }

      inited = true;
    }

    void hardReset() {
      // Zero all internal buffers
      for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < kNumCombs; ++i) {
          std::fill(comb[ch][i].buf.begin(), comb[ch][i].buf.end(), 0.0f);
          comb[ch][i].idx = 0;
          comb[ch][i].filterStore = 0.0f;
        }
        for (int i = 0; i < kNumAP; ++i) {
          std::fill(ap[ch][i].buf.begin(), ap[ch][i].buf.end(), 0.0f);
          ap[ch][i].idx = 0;
        }
      }
      hp1.reset(); hp2.reset();
      modL.reset(); modR.reset();
    }

    void updateDamping(float d01) {
      float d = std::max(0.0f, std::min(1.0f, d01));
      for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < kNumCombs; ++i) {
          comb[ch][i].damp2 = d;
          comb[ch][i].damp1 = 1.0f - d;
        }
      }
    }

    void updateFeedbackForT60(float t60) {
      float t = std::max(0.05f, t60);
      for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < kNumCombs; ++i) {
          const int M = std::max(1, (int)comb[ch][i].buf.size());
          const double g = std::pow(10.0, -3.0 * double(M) / (double(t) * double(kSR)));
          comb[ch][i].feedback = (float)g;
        }
      }
    }

    inline float procComb(Comb &C, float inp) {
      const float y = C.buf[C.idx];
      C.filterStore = (y * C.damp1) + (C.filterStore * C.damp2);
      const float v = inp + C.filterStore * C.feedback;
      C.buf[C.idx] = v;
      C.idx++; if (C.idx >= (int)C.buf.size()) C.idx = 0;
      return y;
    }

    inline float procAllpass(Allpass &A, float x) {
      const float bufout = A.buf[A.idx];
      const float y = -x + bufout;
      A.buf[A.idx] = x + bufout * kAPFeedback;
      A.idx++; if (A.idx >= (int)A.buf.size()) A.idx = 0;
      return y;
    }

    void process(const float* x, float* outLR, size_t n) {
      if (!inited) init();

      // Reset on epoch bump (e.g., bypass edge)
      uint64_t curEpoch = gReverbResetEpoch.load(std::memory_order_acquire);
      if (curEpoch != lastResetSeen) {
        hardReset();
        lastResetSeen = curEpoch;
      }

      // Pull latest UI parameters
      const float mix01_ui    = clampf(RvbMixStore().load(std::memory_order_relaxed) * 0.01f, 0.0f, 1.0f);
      const float t60_ui      = clampf(RvbT60Store().load(std::memory_order_relaxed), 0.05f, rvb_max_decay_sec);
      const float damp01_ui   = clampf(RvbDampStore().load(std::memory_order_relaxed), 0.0f, 1.0f);
      const float width_ui    = clampf(RvbWidthStore().load(std::memory_order_relaxed), 0.0f, 1.0f);
      const float lcHz_ui     = clampf(RvbLCHzStore().load(std::memory_order_relaxed), 10.0f, rvb_lc_max_hz);
      const float modRate_ui  = clampf(RvbModRateStore().load(std::memory_order_relaxed), 0.05f, rvb_mod_max_rate_hz);
      const float modDepth_ui = clampf(RvbModDepthMsStore().load(std::memory_order_relaxed), 0.0f, rvb_mod_max_depth_ms);

      // Update params if changed
      if (std::fabs(gKnobs.rvb_damp.load() - damp01_ui) > 1e-4f) {
        gKnobs.rvb_damp.store(damp01_ui);
        updateDamping(damp01_ui);
        prevDmp = damp01_ui;
      }
      if (std::fabs(gKnobs.rvb_decay.load() - t60_ui) > 1e-4f) {
        gKnobs.rvb_decay.store(t60_ui);
        updateFeedbackForT60(t60_ui);
        prevT60 = t60_ui;
      }
      if (std::fabs(prevLcHz - lcHz_ui) > 1e-3f) {
        hp1.setCutoff(lcHz_ui, kSR);
        hp2.setCutoff(lcHz_ui, kSR);
        prevLcHz = lcHz_ui;
      }
      if (std::fabs(prevModRate - modRate_ui) > 1e-5f || std::fabs(prevModDepth - modDepth_ui) > 1e-4f) {
        // Keep stereo decorrelation via different base delays and phase offset
        modL.setParams(modRate_ui, modDepth_ui, kSR, 12.0f, 0.0f);
        modR.setParams(modRate_ui, modDepth_ui, kSR, 13.7f, float(M_PI)*0.5f);
        prevModRate  = modRate_ui;
        prevModDepth = modDepth_ui;
      }
      gKnobs.rvb_width.store(width_ui);

      const float wet = std::max(0.0f, std::min(1.0f, mix01_ui));
      const float dry = 1.0f - wet;
      const float w   = std::max(0.0f, std::min(1.0f, width_ui));
      const float wet1 = wet * (w * 0.5f + 0.5f);
      const float wet2 = wet * ((1.0f - w) * 0.5f);

      for (size_t i = 0; i < n; ++i) {
        const float inDry = x[i];

        // Pre-tank low-cut on wet only
        float xhp = hp2.process(hp1.process(inDry));
        const float inToTank = xhp * kFixedGain;

        float sumL = 0.0f, sumR = 0.0f;
        for (int c = 0; c < kNumCombs; ++c) {
          sumL += procComb(comb[0][c], inToTank);
          sumR += procComb(comb[1][c], inToTank);
        }

        float outL = sumL, outR = sumR;
        for (int a = 0; a < kNumAP; ++a) {
          outL = procAllpass(ap[0][a], outL);
          outR = procAllpass(ap[1][a], outR);
        }

        // Post-tank modulation (wet only)
        float wetL = modL.tick(outL);
        float wetR = modR.tick(outR);

        const float L = wetL * wet1 + wetR * wet2 + dry * inDry;
        const float R = wetR * wet1 + wetL * wet2 + dry * inDry;

        outLR[2*i + 0] = L;
        outLR[2*i + 1] = R;
      }
    }
  };

  static Freeverb48Stereo R;
  if (prewarmOnly) {           // allocate/initialise from main() before the stream starts
    if (!R.inited) R.init();
    return;
  }
  // Hard bypass when reverb is disabled: copy mono to L/R and return.
  if (RvbEnabledStore().load(std::memory_order_relaxed) == 0) {
    for (size_t i = 0; i < n; ++i) {
      float v = inMono[i];
      outLRInterleaved[2*i + 0] = v;
      outLRInterleaved[2*i + 1] = v;
    }
    return;
  }
  if (n == 0) return;
  R.process(inMono, outLRInterleaved, n);
}

static void ReverbReset() {
  // Bump reset epoch; the processor will zero internal buffers on next call.
  gReverbResetEpoch.fetch_add(1, std::memory_order_acq_rel);
}
