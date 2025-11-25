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
#include <iomanip>
#include <portaudio.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>

// TensorFlow Lite
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#ifdef TFLITE_WITH_XNNPACK
  #include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#endif

// ───────────────────────── GLOBALS/FLAGS ─────────────────────────
// Rolling window of style prob (weighted by coreLen)
static constexpr float windowSec = 2.0;
static constexpr float kPickAttackBaselineBoostDB = 22.0f;
static constexpr float kPickResoBaselineBoostDB   = 32.0f;

std::atomic<bool> gStopFlag(false), gBypass(false);
std::atomic<uint64_t> gBypassEpoch{0};
std::atomic<uint64_t> gReverbResetEpoch{0};

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

const double MAX_PICK_ATTACK_KNOB_CHANGE = 24.0f;
const double MAX_PICK_RESO_KNOB_CHANGE = 24.0f;
const double MAX_IR_KNOB_CHANGE = 24.0f;
const double MAX_MAIN_KNOB_CHANGE = 24.0f;

// UI knobs (atomic so audio thread reads lock-free)
struct Knobs {
  std::atomic<float> main_dB{0.0f};
  std::atomic<float> pickAttack_dB{0.0f};
  std::atomic<float> pickReso_dB{0.0f};
  std::atomic<float> ir_dB{0.0f};
  std::atomic<float> wet_pct{100.0f}; // 0..100

  // Output gain (dB → converted to linear for TFLite input)
  std::atomic<float> outGain_dB{-3.0f};

  // Phase invert switch: 0.0 = normal, 1.0 = inverted
  std::atomic<int> phaseInvert01{0};

  // Noise Gate
  std::atomic<int> ng_enabled{0};
  std::atomic<float> ng_threshold_db{-70.0f};
  std::atomic<float> ng_hysteresis_db{6.0f};
  std::atomic<float> ng_range_db{48.0f};
  std::atomic<float> ng_attack_ms{2.0f};
  std::atomic<float> ng_release_ms{150.0f};
  std::atomic<float> ng_hold_ms{400.0f};
  std::atomic<float> ng_sc_hpf_hz{150.0f};

  // Feedback Protection (frequency shifter)
  std::atomic<int>   afb_enabled{0};   // 0/1 (OFF by default)
  std::atomic<float> afb_shift_hz{5.0f}; // Hz

  // Low-cut (high-pass)
  std::atomic<float> lc_hz{20.0f};
  std::atomic<float> lc_slope_dbpoct{12.0f};

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

  // Reverb
  std::atomic<int> rvb_enabled{0};
  std::atomic<float> rvb_mix{35.0f};
  std::atomic<float> rvb_decay{6.20f};
  std::atomic<float> rvb_damp{0.30f};
  std::atomic<float> rvb_width{0.70f};
  std::atomic<float> rvb_lc_hz{250.0f};
  std::atomic<float> rvb_mod_rate{0.30f};
  std::atomic<float> rvb_mod_depth{1.0f};

  // Delay
  std::atomic<int>   dly_enabled{0};            // 0/1
  std::atomic<float> dly_mix{18.0f};            // % 0..100
  std::atomic<float> dly_time_ms{350.0f};       // ms
  std::atomic<float> dly_feedback{35.0f};       // % 0..100
  std::atomic<float> dly_highcut_hz{4500.0f};   // Hz
  std::atomic<float> dly_lowcut_hz{120.0f};     // Hz (NEW)
  std::atomic<float> dly_width{80.0f};          // % 0..100
  std::atomic<int>   dly_pingpong{0};           // 0/1
  std::atomic<float> dly_mod_rate_hz{0.30f};    // Hz
  std::atomic<float> dly_mod_depth_ms{3.0f};    // ms
  std::atomic<float> dly_drive_db{4.0f};        // dB

  // Stereo Chorus
  std::atomic<int>   ch_enabled{0};        // 0/1 (OFF by default)
  std::atomic<float> ch_mix{25.0f};        // % 0..100
  std::atomic<float> ch_rate_hz{0.80f};    // Hz
  std::atomic<float> ch_depth_ms{6.0f};    // ms
  std::atomic<float> ch_delay_ms{12.0f};   // ms (base delay)
  std::atomic<float> ch_width{80.0f};      // % 0..100
  std::atomic<float> ch_feedback{6.0f};    // % 0..40
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
      float m = std::sinf(phase);
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
                                      const PaStreamCallbackTimeInfo* /*timeInfo*/)
{
  // Hard bypass when reverb is disabled: copy mono to L/R and return.
  if (RvbEnabledStore().load(std::memory_order_relaxed) == 0) {
    for (size_t i = 0; i < n; ++i) {
      float v = inMono[i];
      outLRInterleaved[2*i + 0] = v;
      outLRInterleaved[2*i + 1] = v;
    }
    return;
  }

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
  if (n == 0) return;
  R.process(inMono, outLRInterleaved, n);
}

static void ReverbReset() {
  // Bump reset epoch; the processor will zero internal buffers on next call.
  gReverbResetEpoch.fetch_add(1, std::memory_order_acq_rel);
}














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
static inline float clampf_pos(float x, float hi) { return std::max(0.0f, std::min(hi, x)); }
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
    float a = 1.0f - std::expf(-1.0f / (t * std::max(1.0f, sr)));
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
    float x = std::expf(-2.0f * float(M_PI) * f / srSafe);
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
                                           const PaStreamCallbackTimeInfo* /*timeInfo*/)
{
  if (DlyEnabledStore().load(std::memory_order_relaxed) == 0) {
    return;
  }

  struct TapeDelayStereo {
    struct OnePoleLP {
      float a = 0.0f;
      float y = 0.0f;
      void setCutoff(float fc, float sr) {
        float f = clampf(fc, 20.0f, 20000.0f);
        float srSafe = std::max(1.0f, sr);
        float x = std::expf(-2.0f * float(M_PI) * f / srSafe);
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

        float mL = 0.7f * std::sinf(wowPhaseL) + 0.3f * std::sinf(fltPhaseL);
        float mR = 0.7f * std::sinf(wowPhaseR) + 0.3f * std::sinf(fltPhaseR);

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
  if (n == 0) return;
  D.processAdd(inMono, ioLRInterleaved, n, kSR);
}
























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
                                            const PaStreamCallbackTimeInfo* /*timeInfo*/)
{
  if (ChEnabledStore().load(std::memory_order_relaxed) == 0) return;

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

        float mL = std::sinf(phaseL);
        float mR = std::sinf(phaseR);

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
  if (n == 0) return;
  C.processAdd(inMono, ioLRInterleaved, n, kSR);
}






// Feedback Protection (frequency shifter)

// Limits/defaults for UI
static constexpr float afb_shift_min_hz = 0.5f;
static constexpr float afb_shift_max_hz = 12.0f;

// UI atomics (referencing gKnobs)
static std::atomic<int>&   AFBEnabledStore()     { return gKnobs.afb_enabled; }
static std::atomic<float>& AFBShiftHzStore()     { return gKnobs.afb_shift_hz; }
static std::atomic<uint64_t>& AFBResetEpochRef() { static std::atomic<uint64_t> e{0}; return e; }
static inline void AFBReset() { AFBResetEpochRef().fetch_add(1, std::memory_order_acq_rel); }
static inline uint64_t AFBGetResetEpoch() { return AFBResetEpochRef().load(std::memory_order_acquire); }

struct FIRHilbert90 {
  int N = 0;
  int idx = 0;
  int half = 0;
  std::vector<float> h;
  std::vector<float> z;

  void init(int taps) {
    int T = taps;
    if (T < 7) T = 7;
    if ((T & 1) == 0) T += 1; // ensure odd
    N = T;
    h.resize(N);
    z.assign(N, 0.0f);
    idx = 0;
    half = (N - 1) / 2;

    // Windowed ideal Hilbert: h[k] = 2/(pi*k) for odd k, 0 for even/zero, Hanning window
    for (int n = 0; n < N; ++n) {
      int k = n - half;
      float c = 0.0f;
      if (k != 0 && (k & 1) != 0) {
        c = 2.0f / (float(M_PI) * float(k));
      }
      float w = 0.5f * (1.0f - std::cos(2.0f * float(M_PI) * float(n) / float(N - 1)));
      h[n] = c * w;
    }
  }

  inline float process(float x) {
    z[idx] = x;
    float y = 0.0f;
    int j = idx;
    for (int n = 0; n < N; ++n) {
      y += h[n] * z[j];
      --j; if (j < 0) j = N - 1;
    }
    ++idx; if (idx >= N) idx = 0;
    return y;
  }
};

struct AntiFeedbackShifter {
  FIRHilbert90 hil;
  std::vector<float> dly;
  int dlen = 0;
  int didx = 0;

  float phase = 0.0f;
  float inc   = 0.0f;
  float sr    = kSR;
  bool  inited = false;

  void setSampleRate(float s) { sr = std::max(1000.0f, s); }

  void init() {
    if (inited) return;
    hil.init(63); // small, ~0.65 ms group delay
    dlen = std::max(1, hil.half);
    dly.assign(dlen, 0.0f);
    didx = 0;
    phase = 0.0f;
    inited = true;
  }

  void reset() {
    if (!inited) init();
    std::fill(dly.begin(), dly.end(), 0.0f);
    phase = 0.0f;
  }

  inline float delayProcess(float x) {
    float y = dly[didx];
    dly[didx] = x;
    ++didx; if (didx >= dlen) didx = 0;
    return y;
  }

  void setShiftHz(float hz) {
    float f = clampf(hz, afb_shift_min_hz, afb_shift_max_hz);
    inc = 2.0f * float(M_PI) * (f / std::max(1.0f, sr));
  }

  void processBlock(float* x, size_t n, bool enabled) {
    if (!enabled) return;
    if (!inited) init();
    for (size_t i = 0; i < n; ++i) {
      float xi = x[i];
      float xh = hil.process(xi);   // approx Hilbert{x} delayed by hil.half samples
      float xd = delayProcess(xi);  // delay dry by the same group delay

      float c = std::cos(phase);
      float s = std::sin(phase);
      float y = xd * c - xh * s;    // Re{ (x + j*H{x}) * e^{j wt} }

      x[i] = std::isfinite(y) ? y : 0.0f;

      phase += inc;
      if (phase >= 2.0f * float(M_PI)) phase -= 2.0f * float(M_PI);
    }
  }
};























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
  RingBuffer in, out, dry;
  std::vector<float> tempIn;      // mono input (from ADC ch1)
  std::vector<float> tempOut;     // processed mono
  std::vector<float> tempStereo;  // processed stereo L/R
  std::vector<float> tempDry;     // delayed DI
  explicit Shared(size_t cap) : in(cap), out(cap), dry(cap) {}
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

  // NEW: Main stem gain (realtime_knob_main), dB -> lin
  const float main_lin = db2lin(gKnobs.main_dB.load());
  setIf("realtime_knob_main", main_lin);

  // Output/Phase + Master EQ (all scalar [1,1])
  const float out_lin = db2lin(gKnobs.outGain_dB.load());
  setIf("realtime_knob_outgain", out_lin);
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

  // NEW: Main stem gain (realtime_knob_main), dB -> lin
  const float main_lin = db2lin(gKnobs.main_dB.load());
  setIf("realtime_knob_main", main_lin);

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
#if defined(__linux__)
  struct sched_param sp;
  std::memset(&sp, 0, sizeof(sp));
  sp.sched_priority = 30;  // slightly higher than main
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
      ctx.sh->in .r.store(ctx.sh->in .w.load(std::memory_order_acquire), std::memory_order_release);
      ctx.sh->out.r.store(ctx.sh->out.w.load(std::memory_order_acquire), std::memory_order_release);
      ctx.sh->dry.r.store(ctx.sh->dry.w.load(std::memory_order_acquire), std::memory_order_release);
    }
  };

  // Initial hop
  size_t got = ctx.sh->in.pop(xhop.data(), hop);
  if (got < (size_t)hop) std::fill(xhop.begin()+got, xhop.end(), 0.0f);

  // enqueue dry path with the same hop
  ctx.sh->dry.push(xhop.data(), hop);

  // Helpers to call TFLite for one block
  auto do_invoke_streaming = [&](const float* coreBuf, int coreLen, float guidedStableForAuto){
    float pa_lin = std::pow(10.0f, (gKnobs.pickAttack_dB.load() + kPickAttackBaselineBoostDB) / 20.0f);
    float pr_lin = std::pow(10.0f, (gKnobs.pickReso_dB.load()   + kPickResoBaselineBoostDB)   / 20.0f);
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

      ctx.sh->dry.push(xhop.data(), hop);

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

      ctx.sh->dry.push(xhop.data(), hop);

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
  auto *sh  = static_cast<Shared*>(userData);
  auto *in  = static_cast<const float*>(inputBuffer);
  auto *out = static_cast<float*>(outputBuffer);

  const size_t   hop      = framesPerBuffer;
  const unsigned numInCh  = 8;
  const unsigned numOutCh = 8;

  float *monoIn = sh->tempIn.data();

  // Handle bypass edge + reset time-based DSP
  if (timeInfo) {
    static uint64_t s_lastBypassEpoch = 0;
    static bool     s_lastBypass      = false;

    uint64_t curEpoch  = gBypassEpoch.load(std::memory_order_acquire);
    bool     curBypass = gBypass.load(std::memory_order_relaxed);
    if (curEpoch != s_lastBypassEpoch || curBypass != s_lastBypass) {
      // Flush all rings
      sh->in.r.store (sh->in.w.load (std::memory_order_acquire), std::memory_order_release);
      sh->out.r.store(sh->out.w.load(std::memory_order_acquire), std::memory_order_release);
      sh->dry.r.store(sh->dry.w.load(std::memory_order_acquire), std::memory_order_release);

      s_lastBypassEpoch = curEpoch;
      s_lastBypass      = curBypass;

      // Reset time-based processors
      ReverbReset();
      GateReset();
      DelayReset();
      ChorusReset();
      AFBReset();
    }
  }

  // Deinterleave ADC ch1 (channel 0) into monoIn, even if bypassed
  if (in) {
    for (size_t i = 0; i < hop; ++i) {
      monoIn[i] = in[numInCh * i + 0]; // ADC channel 1
    }
    // Feed mono input to inference thread ring
    sh->in.push(monoIn, hop);
  } else {
    std::memset(monoIn, 0, sizeof(float) * hop);
    sh->in.push(monoIn, hop);
  }

  // Global bypass: route monoIn to outputs
  if (gBypass.load(std::memory_order_relaxed)) {
    // Ensure processed/dry rings are empty while bypassed
    sh->out.r.store(sh->out.w.load(std::memory_order_acquire), std::memory_order_release);
    sh->dry.r.store(sh->dry.w.load(std::memory_order_acquire), std::memory_order_release);

    for (size_t i = 0; i < hop; ++i) {
      float v = monoIn[i];
      out[numOutCh * i + 0] = v; // L
      out[numOutCh * i + 1] = v; // R
      out[numOutCh * i + 2] = v; // DI
      for (unsigned ch = 3; ch < numOutCh; ++ch) {
        out[numOutCh * i + ch] = 0.0f;
      }
    }
    return paContinue;
  }

  // Pop processed mono and delayed dry mono from rings
  float *proc = sh->tempOut.data();
  float *dry  = sh->tempDry.data();
  size_t gotP = sh->out.pop(proc, hop);
  size_t gotD = sh->dry.pop(dry,  hop);
  if (gotP < hop) std::memset(proc + gotP, 0, sizeof(float) * (hop - gotP));
  if (gotD < hop) std::memset(dry  + gotD, 0, sizeof(float) * (hop - gotD));

  // Noise gate on processed mono
  {
    static NoiseGate s_gate;
    static bool s_inited = false;
    if (!s_inited) {
      s_gate.setSampleRate(kSR);
      s_inited = true;
    }
    static uint64_t s_lastGateEpoch = 0;
    uint64_t curGateEpoch = GateGetResetEpoch();
    if (curGateEpoch != s_lastGateEpoch) {
      s_gate.reset();
      s_lastGateEpoch = curGateEpoch;
    }

    const bool gate_enabled    = (GateEnabledStore().load(std::memory_order_relaxed) != 0);
    const float th_db    = GateThreshDBStore().load(std::memory_order_relaxed);
    const float hy_db    = GateHystDBStore().load(std::memory_order_relaxed);
    const float atk_ms   = GateAttackMsStore().load(std::memory_order_relaxed);
    const float rel_ms   = GateReleaseMsStore().load(std::memory_order_relaxed);
    const float hold_ms  = GateHoldMsStore().load(std::memory_order_relaxed);
    const float range_db = GateRangeDBStore().load(std::memory_order_relaxed);
    const float hpf_hz   = GateScHPHzStore().load(std::memory_order_relaxed);

    s_gate.setParams(th_db, hy_db, atk_ms, rel_ms, hold_ms, range_db, hpf_hz);
    s_gate.processBlock(proc, hop, gate_enabled);
  }

  // Anti-feedback shifter on processed mono
  {
    static AntiFeedbackShifter s_afb;
    static bool s_inited = false;
    if (!s_inited) { s_afb.setSampleRate(kSR); s_afb.init(); s_inited = true; }
    static uint64_t s_lastAfbEpoch = 0;
    uint64_t curAfbEpoch = AFBGetResetEpoch();
    if (curAfbEpoch != s_lastAfbEpoch) {
      s_afb.reset();
      s_lastAfbEpoch = curAfbEpoch;
    }
    const bool afb_on = (AFBEnabledStore().load(std::memory_order_relaxed) != 0);
    const float shz   = AFBShiftHzStore().load(std::memory_order_relaxed);
    s_afb.setShiftHz(shz);
    s_afb.processBlock(proc, hop, afb_on);
  }

  const bool rvbOn = (RvbEnabledStore().load(std::memory_order_relaxed) != 0);
  const bool dlyOn = (DlyEnabledStore().load(std::memory_order_relaxed) != 0);

  // Stereo scratch for processed path
  float *st = sh->tempStereo.data();
  std::memset(st, 0, sizeof(float) * 2 * hop);

  if (rvbOn) {
    ReverbProcessMonoToStereo(proc, st, hop, framesPerBuffer, timeInfo);
  } else {
    float dlyMix01 = dlyOn ? clampf(DlyMixStore().load(std::memory_order_relaxed) * 0.01f, 0.0f, 1.0f) : 0.0f;
    float dryAmt   = dlyOn ? (1.0f - dlyMix01) : 1.0f;
    for (size_t i = 0; i < hop; ++i) {
      float v = proc[i] * dryAmt;
      st[2 * i + 0] = v;
      st[2 * i + 1] = v;
    }
  }

  ChorusProcessAddMonoToStereoWet(proc, st, hop, framesPerBuffer, timeInfo);
  DelayProcessAddMonoToStereoWet (proc, st, hop, framesPerBuffer, timeInfo);

  // Write to 8-channel output
  for (size_t i = 0; i < hop; ++i) {
    out[numOutCh * i + 0] = st[2 * i + 0]; // L
    out[numOutCh * i + 1] = st[2 * i + 1]; // R
    out[numOutCh * i + 2] = dry[i];        // DI
    for (unsigned ch = 3; ch < numOutCh; ++ch) {
      out[numOutCh * i + ch] = 0.0f;
    }
  }

  return paContinue;
}

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

  #ifdef __linux__
    // Lock current and future memory to avoid page faults in RT audio threads
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall");
    }

    // Try to give the process a SCHED_FIFO policy (requires CAP_SYS_NICE / root)
    struct sched_param sp;
    std::memset(&sp, 0, sizeof(sp));
    sp.sched_priority = 20;  // tune as needed (1–99)
    if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0) {
      perror("sched_setscheduler");
    }
  #endif

  PaError err = Pa_Initialize();
  if (err != paNoError) {
    std::cerr<<"Pa_Initialize failed: "<<Pa_GetErrorText(err)<<"\n";
    return 1;
  }
  auto *diIn  = Pa_GetDeviceInfo(inDevId);
  auto *diOut = Pa_GetDeviceInfo(outDevId);
  int sampleRate = 48000;

  std::cout<<"[MAIN] In ["<<inDevId<<"] "<<diIn->name<<" sr="<<diIn->defaultSampleRate<<"\n";
  std::cout<<"[MAIN] Out["<<outDevId<<"] "<<diOut->name<<" sr="<<diOut->defaultSampleRate<<"\n";
  std::cout<<"[MAIN] blockSize="<<blockSize<<"\n";
  std::cout<<"[MAIN] model: "<<tflitePath<<"\n";

  // PortAudio stream params
  double wantLatencySecs = double(blockSize) / sampleRate;
  PaStreamParameters ip = { inDevId, 8, paFloat32, wantLatencySecs, nullptr };
  PaStreamParameters op = { outDevId, 8, paFloat32, wantLatencySecs, nullptr };

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
shared.tempIn.resize(blockSize);
shared.tempOut.resize(blockSize);
shared.tempStereo.resize(blockSize * 2);
shared.tempDry.resize(blockSize);

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
        shared.in.r .store(shared.in .w.load(std::memory_order_acquire), std::memory_order_release);
        shared.out.r.store(shared.out.w.load(std::memory_order_acquire), std::memory_order_release);
        shared.dry.r.store(shared.dry.w.load(std::memory_order_acquire), std::memory_order_release);

        std::cout<<"[UI] Bypass "<<(now?"ON":"OFF")<<"\n";
      } else if (c=='q') { gKnobs.pickAttack_dB.store( clampf(gKnobs.pickAttack_dB.load()+1.0f, -MAX_PICK_RESO_KNOB_CHANGE, MAX_PICK_RESO_KNOB_CHANGE) ); }
        else if (c=='a') { gKnobs.pickAttack_dB.store( clampf(gKnobs.pickAttack_dB.load()-1.0f, -MAX_PICK_RESO_KNOB_CHANGE, MAX_PICK_RESO_KNOB_CHANGE) ); }
        else if (c=='w') { gKnobs.pickReso_dB.store( clampf(gKnobs.pickReso_dB.load()+1.0f, -MAX_PICK_RESO_KNOB_CHANGE, MAX_PICK_RESO_KNOB_CHANGE) ); }
        else if (c=='s') { gKnobs.pickReso_dB.store( clampf(gKnobs.pickReso_dB.load()-1.0f, -MAX_PICK_RESO_KNOB_CHANGE, MAX_PICK_RESO_KNOB_CHANGE) ); }
        else if (c=='e') { gKnobs.ir_dB.store( clampf(gKnobs.ir_dB.load()+1.0f, -MAX_PICK_RESO_KNOB_CHANGE, MAX_PICK_RESO_KNOB_CHANGE) ); }
        else if (c=='d') { gKnobs.ir_dB.store( clampf(gKnobs.ir_dB.load()-1.0f, -MAX_PICK_RESO_KNOB_CHANGE, MAX_PICK_RESO_KNOB_CHANGE) ); }
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

  // Comment out for now
  // StartKnobWebUI(8088);

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

  // Comment out for now
  // StopKnobWebUI();

  Pa_StopStream(stream);
  Pa_CloseStream(stream);
  Pa_Terminate();

  infThread.join();
  kbThread.join();
  tcsetattr(STDIN_FILENO,TCSANOW,&oldt);
  std::cout<<"[MAIN] Done.\n";
  return 0;
}