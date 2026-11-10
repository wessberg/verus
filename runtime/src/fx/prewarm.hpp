#pragma once
// Touches every effect once so first use on the RT thread does no lazy initialisation.

#include "common.hpp"
#include "fx/anti_feedback.hpp"
#include "fx/chorus.hpp"
#include "fx/master_eq.hpp"

static void PrewarmDsp() {
  ReverbProcessMonoToStereo(nullptr, nullptr, 0, 0, nullptr, true);
  DelayProcessAddMonoToStereoWet(nullptr, nullptr, 0, 0, nullptr, true);
  ChorusProcessAddMonoToStereoWet(nullptr, nullptr, 0, 0, nullptr, true);
  gGate.setSampleRate(kSR);
  gAfb.setSampleRate(kSR);
  gAfb.init();
  gMasterEq.sr = kSR;
  gMasterEq.updateFromKnobs();
}
