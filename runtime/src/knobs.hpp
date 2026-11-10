#pragma once
// Runtime controls (atomic, so the audio thread reads them lock-free).

#include "common.hpp"

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

  // Pickup type fed to the model: 0.0 = piezo, 1.0 = magnetic (electric guitars)
  std::atomic<float> pickup01{0.0f};

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

  // Master EQ — a runtime IIR effect applied post-model (MasterEqIIR below), NOT a model input
  // any more. Defaults are the neutral position (LC at 20 Hz = bypass, all gains 0 dB) and
  // mirror master_eq_defaults() in src/verus/eq.py; `--process --eq ...` uses the same formulas.
  std::atomic<float> lc_hz{20.0f};
  std::atomic<float> lc_slope_dbpoct{6.0f};

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
