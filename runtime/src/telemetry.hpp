#pragma once
// Counters the RT threads write and main() prints.

#include "common.hpp"

// ───────────────────────── Telemetry (RT threads write atomics, main() prints) ─────────────
// No std::cout/std::cerr from the inference thread or the PortAudio callback: terminal I/O can
// block for milliseconds (SSH, scrollback), which is an xrun at a 0.67 ms block period.
struct Telemetry {
  std::atomic<double>   procMsAvg{0.0};
  std::atomic<double>   predWindow{0.5};
  std::atomic<double>   dispP{0.5};
  std::atomic<uint64_t> periodicSeq{0};   // bumped every PROCESSING_LOGGING_WAIT_SEC by the inference thread
  std::atomic<uint64_t> knobSeq{0};       // bumped when a knob/mode changed
  std::atomic<bool>     streaming{true};
  std::atomic<uint32_t> outUnderruns{0};  // callback found < hop processed samples (xrun)
  std::atomic<uint32_t> inDrops{0};       // input ring full: inference thread fell behind
  std::atomic<uint32_t> resyncs{0};       // backlog trimmed to keep latency at one hop
  std::atomic<uint32_t> paXruns{0};       // PortAudio reported input overflow / output underflow
  std::atomic<uint32_t> badBlocks{0};     // callback got framesPerBuffer != blockSize (silenced)
  std::atomic<uint32_t> invokeFails{0};   // TFLite Invoke() errors (output silenced for that hop)
  std::atomic<bool>     rtPrioFailed{false};
};
static Telemetry gTelemetry;
