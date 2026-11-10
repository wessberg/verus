#pragma once
// Raspberry Pi SoC temperature monitor.

#include "common.hpp"
#include "state.hpp"

// ───────────────────────── Raspberry Pi temperature monitor ─────────────────────────
// Don't call `vcgencmd` from RT threads (fork/exec). Read sysfs once/sec in a low-priority thread.
static std::atomic<float> gSoCTempC{std::numeric_limits<float>::quiet_NaN()};
static std::atomic<bool>  gTempMonStop{false};
static std::thread        gTempMonThread;

static inline float ReadSoCTempC() {
#if defined(__linux__)
  // millidegrees C, e.g. "45123\n"
  std::ifstream f("/sys/class/thermal/thermal_zone0/temp");
  long milli = 0;
  if (f && (f >> milli)) return float(milli) * 0.001f;
#endif
  return std::numeric_limits<float>::quiet_NaN();
}

static inline void AppendSoCTemp(std::ostream& os) {
  float t = gSoCTempC.load(std::memory_order_relaxed);
  if (std::isfinite(t)) {
    auto oldFlags = os.flags();
    auto oldPrec  = os.precision();
    os << "  Temp=" << std::fixed << std::setprecision(1) << t << "C";
    os.flags(oldFlags);
    os.precision(oldPrec);
  } else {
    os << "  Temp=n/a";
  }
}

static void StartTempMonitor(int periodMs = 1000) {
  if (gTempMonThread.joinable()) return;
  gTempMonStop.store(false, std::memory_order_relaxed);

  // Seed an initial value
  gSoCTempC.store(ReadSoCTempC(), std::memory_order_relaxed);

  gTempMonThread = std::thread([periodMs]{
#if defined(__linux__)
    // If main() switched to SCHED_FIFO, std::thread inherits it. Force non-RT here.
    sched_param sp{}; sp.sched_priority = 0;
    (void)pthread_setschedparam(pthread_self(), SCHED_OTHER, &sp);
#endif
    while (!gTempMonStop.load(std::memory_order_relaxed) &&
           !gStopFlag.load(std::memory_order_relaxed)) {
      gSoCTempC.store(ReadSoCTempC(), std::memory_order_relaxed);
      std::this_thread::sleep_for(std::chrono::milliseconds(periodMs));
    }
  });
}

static void StopTempMonitor() {
  gTempMonStop.store(true, std::memory_order_relaxed);
  if (gTempMonThread.joinable()) gTempMonThread.join();
}
