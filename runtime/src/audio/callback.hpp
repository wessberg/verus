#pragma once
// The PortAudio callback: I/O routing, FX chain, monitoring.

#include "common.hpp"
#include "audio/ring_buffer.hpp"
#include "fx/anti_feedback.hpp"
#include "fx/chorus.hpp"
#include "fx/master_eq.hpp"
#include "realtime.hpp"
#include "telemetry.hpp"

// ───────────────────────── PA CALLBACK ─────────────────────────

static int paCallback(const void *inputBuffer,
                      void *outputBuffer,
                      unsigned long framesPerBuffer,
                      const PaStreamCallbackTimeInfo *timeInfo,
                      PaStreamCallbackFlags statusFlags,
                      void *userData)
{
  auto *sh  = static_cast<Shared*>(userData);
  auto *in  = static_cast<const float*>(inputBuffer);
  auto *out = static_cast<float*>(outputBuffer);

  const unsigned numInCh  = 8;
  const unsigned numOutCh = 8;

  // One-time RT setup of the PortAudio callback thread (it inherits main()'s policy otherwise,
  // which left it BELOW the inference thread): explicit SCHED_FIFO priority + flush-to-zero.
  {
    static bool s_threadSetup = false;
    if (!s_threadSetup) {
      s_threadSetup = true;
      if (!setThreadRealtime(kCallbackRtPriority))
        gTelemetry.rtPrioFailed.store(true, std::memory_order_relaxed);
      enableFlushToZero();
    }
  }

  if (statusFlags & (paInputOverflow | paOutputUnderflow))
    gTelemetry.paXruns.fetch_add(1, std::memory_order_relaxed);

  // Every scratch buffer is sized for exactly `blockSize` frames and hop == T is enforced; if the
  // host ever delivers a different block size, output silence instead of overrunning buffers.
  if (framesPerBuffer != sh->hop) {
    if (out) std::memset(out, 0, sizeof(float) * numOutCh * framesPerBuffer);
    gTelemetry.badBlocks.fetch_add(1, std::memory_order_relaxed);
    return paContinue;
  }
  const size_t   hop      = framesPerBuffer;

  float *monoIn  = sh->tempIn.data();
  float *fxRetL  = sh->tempFxReturnL.data();
  float *fxRetR  = sh->tempFxReturnR.data();

  // Jitter buffer: after start-up, a flush or an underrun, nothing is popped (silence out) until
  // latencyHops processed hops are queued again, so the configured scheduling headroom is
  // restored instead of being lost for good. Both rings are skipped together → DI stays aligned.
  static bool s_buffering = true;

  // Handle bypass edge + reset time-based DSP. This callback is the ONLY consumer of the out/dry
  // rings (flushes from the keyboard/inference threads raced with pop()).
  {
    static uint64_t s_lastBypassEpoch = 0;
    static bool     s_lastBypass      = false;

    uint64_t curEpoch  = gBypassEpoch.load(std::memory_order_acquire);
    bool     curBypass = gBypass.load(std::memory_order_relaxed);
    if (curEpoch != s_lastBypassEpoch || curBypass != s_lastBypass) {
      // Flush the processed/DI rings (the input ring keeps feeding the model so its streaming
      // states stay warm across bypass)
      dropAligned(sh->out, sh->dry, sh->out.available());
      s_buffering = true;

      s_lastBypassEpoch = curEpoch;
      s_lastBypass      = curBypass;

      // Reset time-based processors
      ReverbReset();
      GateReset();
      DelayReset();
      ChorusReset();
      AFBReset();
      gMasterEq.reset();
    }
  }

  // 1) Pop processed + DI from the rings FIRST — before this block's input is pushed — so the
  //    rings only ever hold blocks from earlier callbacks and the steady-state latency is exactly
  //    one hop (popping after the push would race the inference thread on multi-core).
  float *proc = sh->tempOut.data();
  float *dry  = sh->tempDry.data();
  {
    // Steady state: latencyHops processed hops are queued when the callback runs (the inference
    // thread must finish hop k before callback k+latencyHops). More than that is backlog from an
    // earlier stall: drop whole hops from BOTH rings so DI/processed stay sample-aligned and the
    // latency returns to latencyHops instead of creeping up by a hop per xrun.
    const size_t L = sh->latencyHops;
    const size_t availP = sh->out.available();
    if (availP >= (L + 1) * hop) {
      size_t excess = availP - L * hop;
      excess -= excess % hop;
      dropAligned(sh->out, sh->dry, excess);
      gTelemetry.resyncs.fetch_add(1, std::memory_order_relaxed);
    }
  }
  if (s_buffering && sh->out.available() >= sh->latencyHops * hop) s_buffering = false;
  size_t gotP = 0, gotD = 0;
  if (!s_buffering) {
    gotP = sh->out.pop(proc, hop);
    gotD = sh->dry.pop(dry, gotP);                     // same count as processed → stays aligned
    if (gotP < hop) {
      gTelemetry.outUnderruns.fetch_add(1, std::memory_order_relaxed);
      s_buffering = true;                              // re-establish the headroom before resuming
    }
  }
  if (gotP < hop) std::memset(proc + gotP, 0, sizeof(float) * (hop - gotP));
  if (gotD < hop) std::memset(dry + gotD, 0, sizeof(float) * (hop - gotD));

  // Model output guard: a NaN/Inf (or a wild sample) must never reach the stateful FX below —
  // the reverb/delay feedback loops would latch it forever. (The build uses -ffast-math; see
  // CMakeLists.txt for why -fno-finite-math-only keeps std::isfinite meaningful.)
  for (size_t i = 0; i < hop; ++i) {
    const float v = proc[i];
    proc[i] = std::isfinite(v) ? clampf(v, -kProcSampleLimit, kProcSampleLimit) : 0.0f;
  }

  // 2) Deinterleave ADC channels:
  //  - ch1 (index 0): guitar input -> monoIn
  //  - ch3 (index 2): FX RETURN L  -> fxRetL
  //  - ch4 (index 3): FX RETURN R  -> fxRetR
  if (in) {
    for (size_t i = 0; i < hop; ++i) {
      monoIn[i] = in[numInCh * i + 0]; // ADC ch1: guitar in
      fxRetL[i] = in[numInCh * i + 2]; // ADC ch3: FX return L
      fxRetR[i] = in[numInCh * i + 3]; // ADC ch4: FX return R
    }
  } else {
    std::memset(monoIn, 0, sizeof(float) * hop);
    std::memset(fxRetL, 0, sizeof(float) * hop);
    std::memset(fxRetR, 0, sizeof(float) * hop);
  }
  // Feed mono input to the inference thread and wake it
  if (sh->in.push(monoIn, hop) < hop) gTelemetry.inDrops.fetch_add(1, std::memory_order_relaxed);
  inputSignalPost();

  // Global bypass: route monoIn to outputs
  if (gBypass.load(std::memory_order_relaxed)) {
    // Keep the processed/dry rings drained while bypassed (the model keeps running)
    dropAligned(sh->out, sh->dry, sh->out.available());
    s_buffering = true;
    for (size_t i = 0; i < hop; ++i) {
      float v = monoIn[i];
      size_t idx = numOutCh * i;

      out[idx + 0] = v; // Main L (dry)
      out[idx + 1] = v; // Main R (dry)
      out[idx + 2] = v; // DI (dry, time-alignment irrelevant in bypass)
      out[idx + 3] = v; // FX SEND L (dry)
      out[idx + 4] = v; // FX SEND R (dry)
      out[idx + 5] = v; // Headphone L (dry)
      out[idx + 6] = v; // Headphone R (dry)
      out[idx + 7] = v; // MONO OUT (dry; MAIN L/R are both v anyway)
    }
    return paContinue;
  }

  // Master EQ on processed mono (post-model; the model itself is EQ-free)
  gMasterEq.updateFromKnobs();
  gMasterEq.processBlock(proc, hop);

  // Noise gate on processed mono
  {
    static uint64_t s_lastGateEpoch = 0;
    uint64_t curGateEpoch = GateGetResetEpoch();
    if (curGateEpoch != s_lastGateEpoch) {
      gGate.reset();
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

    gGate.setParams(th_db, hy_db, atk_ms, rel_ms, hold_ms, range_db, hpf_hz);
    gGate.processBlock(proc, hop, gate_enabled);
  }

  // Anti-feedback shifter on processed mono
  {
    static uint64_t s_lastAfbEpoch = 0;
    uint64_t curAfbEpoch = AFBGetResetEpoch();
    if (curAfbEpoch != s_lastAfbEpoch) {
      gAfb.reset();
      s_lastAfbEpoch = curAfbEpoch;
    }
    const bool afb_on = (AFBEnabledStore().load(std::memory_order_relaxed) != 0);
    const float shz   = AFBShiftHzStore().load(std::memory_order_relaxed);
    gAfb.setShiftHz(shz);
    gAfb.processBlock(proc, hop, afb_on);
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

  // Final routing:
  //  - preFxL/preFxR: internal processed stereo (after reverb/delay/chorus)
  //  - FX SEND (DAC ch4/5): always preFxL/preFxR
  //  - FX RETURN (ADC ch3/4): fxRetL/fxRetR
  //  - Main OUT + Headphones: from FX RETURN if loop enabled, else from preFx
  const bool fxOn = gFxLoopEnabled.load(std::memory_order_relaxed);

    for (size_t i = 0; i < hop; ++i) {
    float preFxL = st[2 * i + 0];
    float preFxR = st[2 * i + 1];

    float sendL  = preFxL;
    float sendR  = preFxR;

    float retL   = fxRetL[i];
    float retR   = fxRetR[i];

    float mainL, mainR;
    if (fxOn) {
      mainL = retL;
      mainR = retR;
    } else {
      mainL = preFxL;
      mainR = preFxR;
    }

    float hpL = mainL;
    float hpR = mainR;

    // Mono sum of MAIN L/R (−6 dB to keep level consistent for centered signals)
    float monoMain = 0.5f * (mainL + mainR);

    size_t idx = numOutCh * i;
    out[idx + 0] = mainL;    // DAC ch1: main L
    out[idx + 1] = mainR;    // DAC ch2: main R
    out[idx + 2] = dry[i];   // DAC ch3: time-aligned DI

    out[idx + 3] = sendL;    // DAC ch4: FX SEND L
    out[idx + 4] = sendR;    // DAC ch5: FX SEND R

    out[idx + 5] = hpL;      // DAC ch6: Headphone L
    out[idx + 6] = hpR;      // DAC ch7: Headphone R

    out[idx + 7] = monoMain; // DAC ch8: MONO OUT (MAIN L/R sum)
  }

  return paContinue;
}
