#pragma once
// Tape / modulated stereo delay.

#include "common.hpp"
#include "fx/reverb.hpp"

// ───────────────────────── TAPE/MOD DELAY (Mono in → Stereo wet add) ─────────────────────────
// Features:
//  - Mono input, stereo wet output (added to existing LR).
//  - Wow/flutter modulation of delay time.
//  - Feedback with soft saturation and high-cut low-pass (tape tone).
//  - Stereo width and optional ping-pong (cross feedback).
//  - Proper reset on bypass edges via DelayReset() epoch.
//
// UI stores: enable, mix %, time ms, feedback %, highcut Hz, width %, ping-pong, mod rate Hz, mod depth ms, drive dB.

static constexpr float dly_time_max_ms         = 2000.0f;
static constexpr float dly_mod_max_rate_hz     = 8.0f;
static constexpr float dly_mod_max_depth_ms    = 12.0f;
static constexpr float dly_highcut_min_hz      = 500.0f;
static constexpr float dly_highcut_max_hz      = 20000.0f;
static constexpr float dly_lowcut_min_hz       = 10.0f;
static constexpr float dly_lowcut_max_hz       = 4000.0f; 
static constexpr float dly_max_feedback_pct    = 100.0f;
static constexpr float dly_default_flutter_hz  = 4.5f;
static constexpr float dly_default_stereo_offs_ms = 7.0f;

// Delay UI atomics now refer to gKnobs
static std::atomic<int>&   DlyEnabledStore()     { return gKnobs.dly_enabled; }
static std::atomic<float>& DlyMixStore()         { return gKnobs.dly_mix; }
static std::atomic<float>& DlyTimeMsStore()      { return gKnobs.dly_time_ms; }
static std::atomic<float>& DlyFeedbackStore()    { return gKnobs.dly_feedback; }
static std::atomic<float>& DlyHighcutHzStore()   { return gKnobs.dly_highcut_hz; }
static std::atomic<float>& DlyLowcutHzStore()    { return gKnobs.dly_lowcut_hz; }
static std::atomic<float>& DlyWidthStore()       { return gKnobs.dly_width; }
static std::atomic<int>&   DlyPingPongStore()    { return gKnobs.dly_pingpong; }
static std::atomic<float>& DlyModRateHzStore()   { return gKnobs.dly_mod_rate_hz; }
static std::atomic<float>& DlyModDepthMsStore()  { return gKnobs.dly_mod_depth_ms; }
static std::atomic<float>& DlyDriveDBStore()     { return gKnobs.dly_drive_db; }

static void DelayUISet(bool enabled, float mix_pct, float time_ms, float feedback_pct,
                       float highcut_hz, float lowcut_hz, float width_pct, bool pingpong,
                       float mod_rate_hz, float mod_depth_ms, float drive_db)
{
  DlyEnabledStore().store(enabled ? 1 : 0, std::memory_order_relaxed);
  DlyMixStore().store(clampf(mix_pct, 0.0f, 100.0f), std::memory_order_relaxed);
  DlyTimeMsStore().store(clampf(time_ms, 5.0f, dly_time_max_ms), std::memory_order_relaxed);
  DlyFeedbackStore().store(clampf(feedback_pct, 0.0f, dly_max_feedback_pct), std::memory_order_relaxed);
  DlyHighcutHzStore().store(clampf(highcut_hz, dly_highcut_min_hz, dly_highcut_max_hz), std::memory_order_relaxed);
  DlyLowcutHzStore().store(clampf(lowcut_hz, dly_lowcut_min_hz, dly_lowcut_max_hz), std::memory_order_relaxed);
  DlyWidthStore().store(clampf(width_pct, 0.0f, 100.0f), std::memory_order_relaxed);
  DlyPingPongStore().store(pingpong ? 1 : 0, std::memory_order_relaxed);
  DlyModRateHzStore().store(clampf(mod_rate_hz, 0.01f, dly_mod_max_rate_hz), std::memory_order_relaxed);
  DlyModDepthMsStore().store(clampf(mod_depth_ms, 0.0f, dly_mod_max_depth_ms), std::memory_order_relaxed);
  DlyDriveDBStore().store(clampf(drive_db, 0.0f, 18.0f), std::memory_order_relaxed);
}

static void DelayUIGet(bool &enabled, float &mix_pct, float &time_ms, float &feedback_pct,
                       float &highcut_hz, float &width_pct, bool &pingpong,
                       float &mod_rate_hz, float &mod_depth_ms, float &drive_db,
                       float &lowcut_hz /* NEW */)
{
  enabled       = (DlyEnabledStore().load(std::memory_order_relaxed) != 0);
  mix_pct       = DlyMixStore().load(std::memory_order_relaxed);
  time_ms       = DlyTimeMsStore().load(std::memory_order_relaxed);
  feedback_pct  = DlyFeedbackStore().load(std::memory_order_relaxed);
  highcut_hz    = DlyHighcutHzStore().load(std::memory_order_relaxed);
  width_pct     = DlyWidthStore().load(std::memory_order_relaxed);
  pingpong      = (DlyPingPongStore().load(std::memory_order_relaxed) != 0);
  mod_rate_hz   = DlyModRateHzStore().load(std::memory_order_relaxed);
  mod_depth_ms  = DlyModDepthMsStore().load(std::memory_order_relaxed);
  drive_db      = DlyDriveDBStore().load(std::memory_order_relaxed);
  lowcut_hz     = DlyLowcutHzStore().load(std::memory_order_relaxed);
}

// Reset epoch for delay (for bypass edges)
static std::atomic<uint64_t>& DelayResetEpochRef() {
  static std::atomic<uint64_t> e{0};
  return e;
}
static inline void DelayReset() {
  DelayResetEpochRef().fetch_add(1, std::memory_order_acq_rel);
}

// One-pole low-pass for feedback tone
struct OnePoleLP {
  float a = 0.0f; // update coefficient
  float y = 0.0f;
  void setCutoff(float fc, float sr) {
    float f = clampf(fc, 20.0f, 20000.0f);
    float srSafe = std::max(1.0f, sr);
    float x = std::exp(-2.0f * float(M_PI) * f / srSafe);
    a = 1.0f - x; // y += a*(x - y)
  }
  inline float process(float x) {
    y += a * (x - y);
    return y;
  }
  void reset() { y = 0.0f; }
};

// Simple ring delay line with fractional read
struct DelayLine {
  std::vector<float> buf;
  int w = 0;
  int mask = 0;

  void init(int minCapacitySamples) {
    int cap = 1;
    while (cap < minCapacitySamples) cap <<= 1;
    buf.assign(cap, 0.0f);
    mask = cap - 1;
    w = 0;
  }
  inline float readLinear(float D) const {
    const float maxD = float(mask - 2);
    float d = D;
    if (d < 1.0f) d = 1.0f;
    if (d > maxD) d = maxD;
    int dInt = int(std::floor(d));
    float frac = d - float(dInt);
    int r0 = (w - dInt) & mask;
    int r1 = (r0 - 1) & mask;
    return buf[r0] * (1.0f - frac) + buf[r1] * frac;
  }
  inline void write(float v) {
    buf[w] = v;
    w = (w + 1) & mask;
  }
  void reset() {
    if (!buf.empty()) std::fill(buf.begin(), buf.end(), 0.0f);
    w = 0;
  }
};

// Soft saturation for feedback path
static inline float softclip_tanh(float x, float driveK) {
  float k = std::max(0.1f, driveK);
  float y = std::tanh(x * k);
  float n = std::tanh(k);
  return n > 1e-6f ? (y / n) : y;
}

static inline float drive_db_to_k(float drive_db) {
  // Map 0..18 dB roughly into multiplier 0.8..4.0
  float lin = db2lin(clampf(drive_db, 0.0f, 24.0f));
  float k = 0.8f + 0.35f * lin; // gentle growth
  if (k < 0.5f) k = 0.5f;
  if (k > 6.0f) k = 6.0f;
  return k;
}

static void DelayProcessAddMonoToStereoWet(const float* inMono,
                                           float* ioLRInterleaved,
                                           size_t n,
                                           unsigned long /*framesPerBuffer*/,
                                           const PaStreamCallbackTimeInfo* /*timeInfo*/,
                                           bool prewarmOnly = false)
{
  struct TapeDelayStereo {
    struct OnePoleLP {
      float a = 0.0f;
      float y = 0.0f;
      void setCutoff(float fc, float sr) {
        float f = clampf(fc, 20.0f, 20000.0f);
        float srSafe = std::max(1.0f, sr);
        float x = std::exp(-2.0f * float(M_PI) * f / srSafe);
        a = 1.0f - x;
      }
      inline float process(float x) {
        y += a * (x - y);
        return y;
      }
      void reset() { y = 0.0f; }
    };

    DelayLine dlL, dlR;

    // Tone for the delayed tap (wet output + feedback)
    OnePoleLP lpfL, lpfR;
    OnePoleHP hpfL, hpfR;

    // NEW: Tone for the write injection (first repeat)
    OnePoleLP inLpfL, inLpfR;
    OnePoleHP inHpfL, inHpfR;

    // LFO phases
    float wowPhaseL = 0.0f, wowPhaseR = float(M_PI)*0.5f;
    float fltPhaseL = 1.3f,  fltPhaseR = float(M_PI);

    float wowInc = 0.0f, fltInc = 0.0f;
    float depthSamp = 0.0f;

    float curBaseL = 12000.0f, curBaseR = 12000.0f;
    float stepBaseL = 0.0f, stepBaseR = 0.0f;

    float feedback = 0.35f;
    float wet = 0.2f;
    float width = 0.8f;
    float driveK = 1.2f;
    float lowcutHz = 120.0f;
    float highcutHz = 4500.0f;
    bool  pingpong = true;

    bool inited = false;
    uint64_t lastResetSeen = 0;

    void init() {
      dlL.init(4 * 48000);
      dlR.init(4 * 48000);
      lpfL.reset(); lpfR.reset();
      hpfL.reset(); hpfR.reset();
      inLpfL.reset(); inLpfR.reset();
      inHpfL.reset(); inHpfR.reset();
      inited = true;
    }

    void hardReset() {
      dlL.reset(); dlR.reset();
      lpfL.reset(); lpfR.reset();
      hpfL.reset(); hpfR.reset();
      inLpfL.reset(); inLpfR.reset();
      inHpfL.reset(); inHpfR.reset();
      wowPhaseL = 0.0f; wowPhaseR = float(M_PI)*0.5f;
      fltPhaseL = 1.3f;  fltPhaseR = float(M_PI);
    }

    void setParamsFromUI(float sr, size_t blockN) {
      float mix_pct = DlyMixStore().load(std::memory_order_relaxed);
      float time_ms = DlyTimeMsStore().load(std::memory_order_relaxed);
      float fb_pct  = DlyFeedbackStore().load(std::memory_order_relaxed);
      float hc_hz   = DlyHighcutHzStore().load(std::memory_order_relaxed);
      float lc_hz   = DlyLowcutHzStore().load(std::memory_order_relaxed);
      float width_p = DlyWidthStore().load(std::memory_order_relaxed);
      bool  ping    = (DlyPingPongStore().load(std::memory_order_relaxed) != 0);
      float rate_hz = DlyModRateHzStore().load(std::memory_order_relaxed);
      float depth_ms= DlyModDepthMsStore().load(std::memory_order_relaxed);
      float drive_db= DlyDriveDBStore().load(std::memory_order_relaxed);

      wet      = clampf(mix_pct * 0.01f, 0.0f, 1.0f);
      feedback = clampf(fb_pct * 0.01f, 0.0f, dly_max_feedback_pct * 0.01f);
      width    = clampf(width_p * 0.01f, 0.0f, 1.0f);
      pingpong = ping;

      float srSafe = std::max(1.0f, sr);
      highcutHz = clampf(hc_hz, dly_highcut_min_hz, dly_highcut_max_hz);
      lowcutHz  = clampf(lc_hz, dly_lowcut_min_hz,  dly_lowcut_max_hz);

      // Set tone for delayed tap (wet + feedback)
      lpfL.setCutoff(highcutHz, srSafe);
      lpfR.setCutoff(highcutHz, srSafe);
      hpfL.setCutoff(lowcutHz,  srSafe);
      hpfR.setCutoff(lowcutHz,  srSafe);

      // Set tone for write injection (first repeat)
      inLpfL.setCutoff(highcutHz, srSafe);
      inLpfR.setCutoff(highcutHz, srSafe);
      inHpfL.setCutoff(lowcutHz,  srSafe);
      inHpfR.setCutoff(lowcutHz,  srSafe);

      float baseTarget = clampf(time_ms, 5.0f, dly_time_max_ms) * 0.001f * srSafe;
      float stereoOffs = dly_default_stereo_offs_ms * 0.001f * srSafe;

      // Bound within ring
      float maxD = float(dlL.mask - 4);
      if (baseTarget > maxD) baseTarget = maxD;
      float baseTargetL = baseTarget;
      float baseTargetR = std::min(maxD, baseTarget + stereoOffs);

      stepBaseL = (baseTargetL - curBaseL) / float(std::max<size_t>(1, blockN));
      stepBaseR = (baseTargetR - curBaseR) / float(std::max<size_t>(1, blockN));

      depthSamp = clampf(depth_ms, 0.0f, dly_mod_max_depth_ms) * 0.001f * srSafe;
      wowInc = 2.0f * float(M_PI) * clampf(rate_hz, 0.01f, dly_mod_max_rate_hz) / srSafe;
      fltInc = 2.0f * float(M_PI) * dly_default_flutter_hz / srSafe;

      driveK = drive_db_to_k(drive_db);
    }

    inline void stepPhases() {
      wowPhaseL += wowInc; if (wowPhaseL >= 2.0f*float(M_PI)) wowPhaseL -= 2.0f*float(M_PI);
      wowPhaseR += wowInc; if (wowPhaseR >= 2.0f*float(M_PI)) wowPhaseR -= 2.0f*float(M_PI);
      fltPhaseL += fltInc; if (fltPhaseL >= 2.0f*float(M_PI)) fltPhaseL -= 2.0f*float(M_PI);
      fltPhaseR += fltInc; if (fltPhaseR >= 2.0f*float(M_PI)) fltPhaseR -= 2.0f*float(M_PI);
    }

    void processAdd(const float* xMono, float* ioLR, size_t n, float sr) {
      if (!inited) init();

      uint64_t curEpoch = DelayResetEpochRef().load(std::memory_order_acquire);
      if (curEpoch != lastResetSeen) {
        hardReset();
        lastResetSeen = curEpoch;
      }

      setParamsFromUI(sr, n);

      const float wet1 = wet * (width * 0.5f + 0.5f);
      const float wet2 = wet * ((1.0f - width) * 0.5f);

      for (size_t i = 0; i < n; ++i) {
        curBaseL += stepBaseL;
        curBaseR += stepBaseR;

        float mL = 0.7f * std::sin(wowPhaseL) + 0.3f * std::sin(fltPhaseL);
        float mR = 0.7f * std::sin(wowPhaseR) + 0.3f * std::sin(fltPhaseR);

        float DL = curBaseL + depthSamp * mL;
        float DR = curBaseR + depthSamp * mR;

        float yL = dlL.readLinear(DL);
        float yR = dlR.readLinear(DR);

        // Tone/drive on the delayed tap
        float yLproc = lpfL.process(hpfL.process( softclip_tanh(yL, driveK) ));
        float yRproc = lpfR.process(hpfR.process( softclip_tanh(yR, driveK) ));

        // Feedback source uses processed taps
        float fbSrcL = pingpong ? yRproc : yLproc;
        float fbSrcR = pingpong ? yLproc : yRproc;

        float xi = xMono[i];

        // NEW: pre-filter the write (first repeat is immediately toned)
        float xiL = inLpfL.process(inHpfL.process(xi));
        float xiR = inLpfR.process(inHpfR.process(xi));

        float writeL = xiL + feedback * fbSrcL;
        float writeR = (pingpong ? 0.0f : xiR) + feedback * fbSrcR;

        dlL.write(writeL);
        dlR.write(writeR);

        // Stereo wet mix (processed taps)
        float L = yLproc * wet1 + yRproc * wet2;
        float R = yRproc * wet1 + yLproc * wet2;

        ioLR[2*i + 0] += std::isfinite(L) ? L : 0.0f;
        ioLR[2*i + 1] += std::isfinite(R) ? R : 0.0f;

        stepPhases();
      }
    }
  };

  static TapeDelayStereo D;
  if (prewarmOnly) {           // 2 x 4 s delay lines: allocate from main(), never in the callback
    if (!D.inited) D.init();
    return;
  }
  if (DlyEnabledStore().load(std::memory_order_relaxed) == 0) return;
  if (n == 0) return;
  D.processAdd(inMono, ioLRInterleaved, n, kSR);
}
