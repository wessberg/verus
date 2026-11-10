#pragma once
// Stereo chorus.

#include "common.hpp"
#include "fx/delay.hpp"

// ───────────────────────── STEREO CHORUS (Mono → Stereo wet add) ─────────────────────────
// Gentle, musical stereo chorus: modulated short delays with stereo phase offset.
// Mono input, stereo wet add. Place before Delay in PA callback.

// Limits/defaults for UI
static constexpr float ch_rate_max_hz        = 5.0f;
static constexpr float ch_depth_max_ms       = 10.0f;
static constexpr float ch_delay_min_ms       = 2.0f;
static constexpr float ch_delay_max_ms       = 30.0f;
static constexpr float ch_feedback_max_pct   = 40.0f;
static constexpr float ch_default_stereo_offs_ms = 2.7f;

// UI atomics (referencing gKnobs)
static std::atomic<int>&   ChEnabledStore()   { return gKnobs.ch_enabled; }
static std::atomic<float>& ChMixStore()       { return gKnobs.ch_mix; }
static std::atomic<float>& ChRateHzStore()    { return gKnobs.ch_rate_hz; }
static std::atomic<float>& ChDepthMsStore()   { return gKnobs.ch_depth_ms; }
static std::atomic<float>& ChDelayMsStore()   { return gKnobs.ch_delay_ms; }
static std::atomic<float>& ChWidthStore()     { return gKnobs.ch_width; }
static std::atomic<float>& ChFeedbackStore()  { return gKnobs.ch_feedback; }

// Reset epoch
static std::atomic<uint64_t>& ChorusResetEpochRef() { static std::atomic<uint64_t> e{0}; return e; }
static inline void ChorusReset() { ChorusResetEpochRef().fetch_add(1, std::memory_order_acq_rel); }

// UI set/get
static void ChorusUISet(bool enabled, float mix_pct, float rate_hz, float depth_ms, float delay_ms, float width_pct, float feedback_pct) {
  ChEnabledStore().store(enabled ? 1 : 0, std::memory_order_relaxed);
  ChMixStore().store( clampf(mix_pct,   0.0f, 100.0f),       std::memory_order_relaxed);
  ChRateHzStore().store(clampf(rate_hz, 0.05f, ch_rate_max_hz), std::memory_order_relaxed);
  ChDepthMsStore().store(clampf(depth_ms, 0.0f, ch_depth_max_ms), std::memory_order_relaxed);
  ChDelayMsStore().store(clampf(delay_ms, ch_delay_min_ms, ch_delay_max_ms), std::memory_order_relaxed);
  ChWidthStore().store(clampf(width_pct, 0.0f, 100.0f),     std::memory_order_relaxed);
  ChFeedbackStore().store(clampf(feedback_pct, 0.0f, ch_feedback_max_pct), std::memory_order_relaxed);
}
static void ChorusUIGet(bool &enabled, float &mix_pct, float &rate_hz, float &depth_ms, float &delay_ms, float &width_pct, float &feedback_pct) {
  enabled      = (ChEnabledStore().load(std::memory_order_relaxed) != 0);
  mix_pct      = ChMixStore().load(std::memory_order_relaxed);
  rate_hz      = ChRateHzStore().load(std::memory_order_relaxed);
  depth_ms     = ChDepthMsStore().load(std::memory_order_relaxed);
  delay_ms     = ChDelayMsStore().load(std::memory_order_relaxed);
  width_pct    = ChWidthStore().load(std::memory_order_relaxed);
  feedback_pct = ChFeedbackStore().load(std::memory_order_relaxed);
}

// DSP: Mono in → add stereo wet onto ioLR
static void ChorusProcessAddMonoToStereoWet(const float* inMono,
                                            float* ioLRInterleaved,
                                            size_t n,
                                            unsigned long /*framesPerBuffer*/,
                                            const PaStreamCallbackTimeInfo* /*timeInfo*/,
                                            bool prewarmOnly = false)
{
  struct StereoChorus {
    DelayLine dlL, dlR;

    float phaseL = 0.0f, phaseR = float(M_PI)*0.5f;
    float phaseInc = 0.0f;

    float curBaseL = 600.0f, curBaseR = 700.0f;
    float stepBaseL = 0.0f, stepBaseR = 0.0f;
    float depthSamp = 0.0f;

    float wet = 0.25f;
    float width = 0.8f;
    float feedback = 0.06f;

    bool inited = false;
    uint64_t lastResetSeen = 0;

    void init() {
      dlL.init(1<<13); // plenty for 30 ms
      dlR.init(1<<13);
      inited = true;
    }
    void hardReset() {
      dlL.reset(); dlR.reset();
      phaseL = 0.0f; phaseR = float(M_PI)*0.5f;
    }
    inline void stepPhases() {
      phaseL += phaseInc; if (phaseL >= 2.0f*float(M_PI)) phaseL -= 2.0f*float(M_PI);
      phaseR += phaseInc; if (phaseR >= 2.0f*float(M_PI)) phaseR -= 2.0f*float(M_PI);
    }

    void setParamsFromUI(float sr, size_t blockN) {
      float mix_pct=ChMixStore().load(std::memory_order_relaxed);
      float rate   =ChRateHzStore().load(std::memory_order_relaxed);
      float depth  =ChDepthMsStore().load(std::memory_order_relaxed);
      float delay  =ChDelayMsStore().load(std::memory_order_relaxed);
      float widthP =ChWidthStore().load(std::memory_order_relaxed);
      float fb_pct =ChFeedbackStore().load(std::memory_order_relaxed);

      wet      = clampf(mix_pct*0.01f, 0.0f, 1.0f);
      width    = clampf(widthP*0.01f,  0.0f, 1.0f);
      feedback = clampf(fb_pct*0.01f, 0.0f, ch_feedback_max_pct*0.01f);

      float srSafe = std::max(1.0f, sr);
      phaseInc = 2.0f * float(M_PI) * clampf(rate, 0.05f, ch_rate_max_hz) / srSafe;
      depthSamp = clampf(depth, 0.0f, ch_depth_max_ms) * 0.001f * srSafe;

      float baseTarget = clampf(delay, ch_delay_min_ms, ch_delay_max_ms) * 0.001f * srSafe;
      float stereoOffs = ch_default_stereo_offs_ms * 0.001f * srSafe;

      float maxD = float(dlL.mask - 4);
      if (baseTarget > maxD) baseTarget = maxD;
      float baseTargetL = baseTarget;
      float baseTargetR = std::min(maxD, baseTarget + stereoOffs);

      stepBaseL = (baseTargetL - curBaseL) / float(std::max<size_t>(1, blockN));
      stepBaseR = (baseTargetR - curBaseR) / float(std::max<size_t>(1, blockN));
    }

    void processAdd(const float* xMono, float* ioLR, size_t n, float sr) {
      if (!inited) init();

      uint64_t curEpoch = ChorusResetEpochRef().load(std::memory_order_acquire);
      if (curEpoch != lastResetSeen) {
        hardReset(); lastResetSeen = curEpoch;
      }

      setParamsFromUI(sr, n);

      const float wet1 = wet * (width * 0.5f + 0.5f);
      const float wet2 = wet * ((1.0f - width) * 0.5f);

      for (size_t i = 0; i < n; ++i) {
        curBaseL += stepBaseL;
        curBaseR += stepBaseR;

        float mL = std::sin(phaseL);
        float mR = std::sin(phaseR);

        float DL = curBaseL + depthSamp * mL;
        float DR = curBaseR + depthSamp * mR;

        float yL = dlL.readLinear(DL);
        float yR = dlR.readLinear(DR);

        float xi = xMono[i];

        float writeL = xi + feedback * yL;
        float writeR = xi + feedback * yR;
        dlL.write(writeL);
        dlR.write(writeR);

        float L = yL * wet1 + yR * wet2;
        float R = yR * wet1 + yL * wet2;

        ioLR[2*i + 0] += std::isfinite(L) ? L : 0.0f;
        ioLR[2*i + 1] += std::isfinite(R) ? R : 0.0f;

        stepPhases();
      }
    }
  };

  static StereoChorus C;
  if (prewarmOnly) {
    if (!C.inited) C.init();
    return;
  }
  if (ChEnabledStore().load(std::memory_order_relaxed) == 0) return;
  if (n == 0) return;
  C.processAdd(inMono, ioLRInterleaved, n, kSR);
}
