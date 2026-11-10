#pragma once
// Process-wide flags, shared constants and the style mode.

#include "common.hpp"

// ───────────────────────── GLOBALS/FLAGS ─────────────────────────
// Rolling window of style prob (weighted by coreLen)
static constexpr float windowSec = 2.0;
// Runtime pick-branch make-up gain applied at "0 dB" knob position. These mirror
// REALTIME_PICK_ATTACK_RUNTIME_BASELINE_DB / REALTIME_PICK_RESO_RUNTIME_BASELINE_DB in
// src/verus/config.py — keep them in sync so the pedal, demos, and --process all sit at the same
// level. NOTE: finetuned models were auditioned with an 8 dB pickreso boost in Python; there
// is no way to detect "finetuned" from the TFLite file, so adjust here if you deploy one.
static constexpr float kPickAttackBaselineBoostDB = 22.0f;
static constexpr float kPickResoBaselineBoostDB   = 32.0f;

std::atomic<bool> gStopFlag(false), gBypass(false);
std::atomic<uint64_t> gBypassEpoch{0};
std::atomic<uint64_t> gReverbResetEpoch{0};
std::atomic<bool> gFxLoopEnabled(false);

void signalHandler(int){ gStopFlag = true; }

// Hard ceiling for a processed sample entering the post-model FX (+12 dBFS; normal output is ≤ 1)
static constexpr float kProcSampleLimit = 4.0f;

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

// For latency logging
static double gIOInputLatencySec  = 0.0;
static double gIOOutputLatencySec = 0.0;
static int    PROCESSING_LOGGING_WAIT_SEC = 2;
