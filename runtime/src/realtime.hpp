#pragma once
// Real-time thread helpers: input-available signal, SCHED_FIFO priority, flush-to-zero.

#include "common.hpp"

// Input-available signal: the PortAudio callback posts once per pushed hop; the inference thread
// blocks on it (with a short timeout) instead of busy-spinning at SCHED_FIFO, which pinned one
// core at 100% and heated the Pi for nothing.
#if defined(__linux__)
static sem_t gInputSem;
static inline void inputSignalInit() { sem_init(&gInputSem, 0, 0); }
static inline void inputSignalPost() { sem_post(&gInputSem); }
static inline void inputSignalWaitMs(int ms) {
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 30))
  // Monotonic clock: a wall-clock step (NTP at boot) cannot stretch or collapse the wait
  timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  ts.tv_nsec += (long)ms * 1000000L;
  while (ts.tv_nsec >= 1000000000L) { ts.tv_nsec -= 1000000000L; ts.tv_sec += 1; }
  sem_clockwait(&gInputSem, CLOCK_MONOTONIC, &ts);
#else
  timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
  ts.tv_nsec += (long)ms * 1000000L;
  while (ts.tv_nsec >= 1000000000L) { ts.tv_nsec -= 1000000000L; ts.tv_sec += 1; }
  sem_timedwait(&gInputSem, &ts);
#endif
}
#else
static inline void inputSignalInit() {}
static inline void inputSignalPost() {}
static inline void inputSignalWaitMs(int) { std::this_thread::sleep_for(std::chrono::microseconds(100)); }
#endif

// ───────────────────────── RT thread helpers ─────────────────────────
// Priorities: PortAudio callback > inference thread > everything else (main/keyboard stay at the
// process default). Both stay below the default PREEMPT_RT IRQ-thread priority (50) so the I2S/DMA
// interrupt threads that feed the callback are never starved.
static constexpr int kCallbackRtPriority  = 40;
static constexpr int kInferenceRtPriority = 35;

static inline bool setThreadRealtime(int priority) {
#if defined(__linux__)
  sched_param sp{}; sp.sched_priority = priority;
  return pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) == 0;
#else
  (void)priority; return false;
#endif
}

// Denormals: long reverb/delay/IIR tails decay into subnormals, which are slow on many CPUs. Flush
// them to zero on the RT threads (idempotent; the -ffast-math startup code may already have).
static inline void enableFlushToZero() {
#if defined(__aarch64__)
  uint64_t fpcr; __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
  fpcr |= (1ull << 24);                                   // FZ
  __asm__ __volatile__("msr fpcr, %0" :: "r"(fpcr));
#elif defined(__x86_64__) || defined(__SSE__)
  unsigned int csr = __builtin_ia32_stmxcsr();
  __builtin_ia32_ldmxcsr(csr | 0x8040u);                  // FTZ | DAZ
#endif
}
