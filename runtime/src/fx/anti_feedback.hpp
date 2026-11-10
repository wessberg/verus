#pragma once
// Feedback protection (frequency shifter).

#include "common.hpp"
#include "fx/noise_gate.hpp"

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

// Mono-path insert processors used by the PortAudio callback (file-scope so they can be
// constructed and warmed up from main() instead of on the first RT callback).
static NoiseGate           gGate;
static AntiFeedbackShifter gAfb;

// Construct/allocate every static DSP processor BEFORE the stream starts, so neither the first
// audio callback nor the first time an effect is switched on ever mallocs at RT priority.
