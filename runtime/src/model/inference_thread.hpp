#pragma once
// The inference thread: pops hops, runs the model, pushes processed + dry audio.

#include "common.hpp"
#include "audio/ring_buffer.hpp"
#include "model/tflite_realtime.hpp"
#include "prob_window.hpp"
#include "realtime.hpp"
#include "state.hpp"
#include "telemetry.hpp"

// ───────────────────────── INFERENCE THREAD ─────────────────────────
struct InferCtx {
  Shared* sh;
  int hop;
  int sampleRate;
  TFLiteRealtime* rt;   // loaded, validated (hop vs T) and warmed up by main() before the stream starts
};

void inferenceThread(InferCtx ctx) {
  if (!setThreadRealtime(kInferenceRtPriority))
    gTelemetry.rtPrioFailed.store(true, std::memory_order_relaxed);   // main() reports it
  enableFlushToZero();

  TFLiteRealtime& rt = *ctx.rt;
  const int T   = rt.T();
  const int hop = ctx.hop;

  std::vector<float> xhop(hop, 0.0f);
  std::vector<float> yT(T, 0.0f);
  float styleP = 0.0f;  // per-hop style output (already EMA-smoothed by the model)

  size_t hopsInWindow = std::max<size_t>(1, (size_t)std::ceil(windowSec * double(ctx.sampleRate) / double(hop)));
  ProbWindow probWin;
  probWin.init(nextPow2(hopsInWindow + 8));

  // Stabilized display probability = blend(avg_windowSec, current)
  const float kStabBlend = 0.7f;  // larger → more weight to long window
  float stable_for_next  = 0.5f;  // used if guided-auto feeds model (one-hop lag)

  // Last seen knob values (a change publishes a telemetry event; main() prints it)
  float prev_pa  = gKnobs.pickAttack_dB.load();
  float prev_pr  = gKnobs.pickReso_dB.load();
  float prev_ir  = gKnobs.ir_dB.load();
  float prev_wet = gKnobs.wet_pct.load();
  float prev_main= gKnobs.main_dB.load();
  int   prev_mode= gStyleMode.load();

  const bool streaming = rt.isStreaming();
  gTelemetry.streaming.store(streaming, std::memory_order_relaxed);
  if (streaming) rt.resetStates();
  // NOTE: states are deliberately NOT reset on bypass edges. The model keeps running on the
  // live input while bypassed (the callback just discards its output), so its states are warm
  // and un-bypassing is seamless; resetting them produced cold-start artefacts instead.

  auto publishKnobChangeIfAny = [&](float dispP){
    float pa  = gKnobs.pickAttack_dB.load();
    float pr  = gKnobs.pickReso_dB.load();
    float ir  = gKnobs.ir_dB.load();
    float wet = gKnobs.wet_pct.load();
    float mn  = gKnobs.main_dB.load();
    int   sm  = gStyleMode.load();
    if (pa != prev_pa || pr != prev_pr || ir != prev_ir || wet != prev_wet || mn != prev_main || sm != prev_mode) {
      gTelemetry.dispP.store(dispP, std::memory_order_relaxed);
      gTelemetry.knobSeq.fetch_add(1, std::memory_order_relaxed);
      prev_pa = pa; prev_pr = pr; prev_ir = ir; prev_wet = wet; prev_main = mn; prev_mode = sm;
    }
  };

  const int left = T - hop;               // windowed models only
  std::vector<float> win(T, 0.0f);

  // Run the model on one hop (streaming) or on the slid window (legacy windowed models).
  auto invokeOne = [&](const float* hopBuf, float guidedStableForAuto){
    // Same baseline boosts on both paths — they must sit at the same level (and match Python).
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
      st_force = 1.0f; // override the model's internal style with the windowSec-stabilized prob
    }

    if (streaming) {
      rt.invokeStreamingBlock(hopBuf, hop, yT.data(), styleP, pa_lin, pr_lin, ir_lin, dryFrac, st_true, st_force);
    } else {
      // Slide window left by hop; append new hop at end (a zero window == legacy first block)
      if (left > 0) std::memmove(win.data(), win.data() + hop, sizeof(float) * (T - hop));
      std::memcpy(win.data() + left, hopBuf, sizeof(float) * hop);
      rt.invokeWindowed(win.data(), yT.data(), styleP, pa_lin, pr_lin, ir_lin, dryFrac, st_true, st_force);
    }
  };

  auto last_report = std::chrono::steady_clock::now();
  double sum_proc = 0.0;
  size_t cnt = 0;

  while (!gStopFlag.load(std::memory_order_relaxed)) {
    // Block until the callback has pushed a hop (bounded wait so gStopFlag is honoured)
    while (!gStopFlag.load(std::memory_order_relaxed) && ctx.sh->in.available() < (size_t)hop)
      inputSignalWaitMs(2);
    if (gStopFlag.load(std::memory_order_relaxed)) break;

    size_t got = ctx.sh->in.pop(xhop.data(), hop);
    if (got < (size_t)hop) std::fill(xhop.begin() + got, xhop.end(), 0.0f);

    auto t0 = std::chrono::steady_clock::now();
    // Use the previous stabilized prob for guided AUTO (one-hop lag)
    invokeOne(xhop.data(), stable_for_next);
    auto t1 = std::chrono::steady_clock::now();

    // Publish DI and processed audio in lockstep, dry FIRST then out: the callback pops
    // min(hop, out.available()) samples from BOTH rings, so they can never drift apart. (The
    // old code pushed dry before running the model and popped the rings independently: one
    // late block permanently offset DI vs processed by a hop.)
    ctx.sh->dry.push(xhop.data(), hop);
    ctx.sh->out.push(streaming ? yT.data() : yT.data() + left, hop);

    // Rolling {windowSec}s window and stabilized display prob
    float pHop = styleP;
    probWin.push(pHop, float(hop));
    float pAvgWindowSec = probWin.mean();
    float dispP = (1.0f - kStabBlend) * pHop + kStabBlend * pAvgWindowSec;
    stable_for_next = dispP;

    publishKnobChangeIfAny(dispP);

    double proc_ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
    sum_proc += proc_ms; ++cnt;
    if (t1 - last_report >= std::chrono::seconds(PROCESSING_LOGGING_WAIT_SEC)) {
      gTelemetry.procMsAvg.store(sum_proc / double(std::max<size_t>(1, cnt)), std::memory_order_relaxed);
      gTelemetry.invokeFails.store(rt.invokeFailures(), std::memory_order_relaxed);
      gTelemetry.predWindow.store(pAvgWindowSec, std::memory_order_relaxed);
      gTelemetry.dispP.store(dispP, std::memory_order_relaxed);
      gTelemetry.periodicSeq.fetch_add(1, std::memory_order_relaxed);
      last_report = t1; sum_proc = 0.0; cnt = 0;
    }
  }
}
