#pragma once
// Status / telemetry printing (main thread only).

#include "common.hpp"
#include "knobs.hpp"
#include "soc_temp.hpp"
#include "telemetry.hpp"

// ───────────────────────── Telemetry printing (main thread only) ─────────────────────────
static inline const char* pickOrFingerLabel(float p, int styleMode) {
  if (styleMode == static_cast<int>(StyleMode::ForcePick))   return "PICK (forced)";
  if (styleMode == static_cast<int>(StyleMode::ForceFinger)) return "FINGER (forced)";
  return (p >= 0.60f) ? "PICK" : "FINGER"; // slightly conservative UI threshold
}

static void printKnobStatus(const char* tag) {
  const int sm = gStyleMode.load();
  const float dispP = float(gTelemetry.dispP.load(std::memory_order_relaxed));
  std::cout << (tag ? tag : "[UI]")
            << " Main=" << gKnobs.main_dB.load() << " dB  "
            << "PA=" << gKnobs.pickAttack_dB.load() << " dB  "
            << "PR=" << gKnobs.pickReso_dB.load()   << " dB  "
            << "IR=" << gKnobs.ir_dB.load()         << " dB  "
            << "Wet="<< gKnobs.wet_pct.load()       << " %  "
            << "Pred P=" << dispP << "  "
            << "Pred=" << pickOrFingerLabel(dispP, sm) << "  "
            << "Mode=" << styleModeName(sm)
            << (gGuidedAuto.load() ? "  (Guided AUTO ON)" : "");
  AppendSoCTemp(std::cout);
  std::cout << "\n";
}

static void printPeriodicTelemetry() {
  const int sm = gStyleMode.load();
  const double avg_proc = gTelemetry.procMsAvg.load(std::memory_order_relaxed);
  const double io_ms    = (gIOInputLatencySec + gIOOutputLatencySec) * 1000.0;
  const float  pAvg     = float(gTelemetry.predWindow.load(std::memory_order_relaxed));
  const float  dispP    = float(gTelemetry.dispP.load(std::memory_order_relaxed));
  std::cerr << (gTelemetry.streaming.load() ? "[inference/stream] " : "[inference/window] ")
            << "Proc=" << avg_proc << " ms  I/O=" << io_ms << " ms  Total~" << (avg_proc + io_ms) << " ms"
            << "  Pred" << windowSec << "s=" << pAvg << "  DispP=" << dispP
            << "  pred=" << pickOrFingerLabel(dispP, sm) << "  mode=" << styleModeName(sm)
            << (gGuidedAuto.load() ? "  (Guided)" : "")
            << "  xruns=" << gTelemetry.outUnderruns.load(std::memory_order_relaxed)
            << " paXruns=" << gTelemetry.paXruns.load(std::memory_order_relaxed)
            << " inDrops=" << gTelemetry.inDrops.load(std::memory_order_relaxed)
            << " resyncs=" << gTelemetry.resyncs.load(std::memory_order_relaxed);
  const uint32_t bad = gTelemetry.badBlocks.load(std::memory_order_relaxed);
  if (bad) std::cerr << " badBlocks=" << bad;
  const uint32_t invf = gTelemetry.invokeFails.load(std::memory_order_relaxed);
  if (invf) std::cerr << " invokeFails=" << invf;
  AppendSoCTemp(std::cerr);
  std::cerr << "\n";
}
