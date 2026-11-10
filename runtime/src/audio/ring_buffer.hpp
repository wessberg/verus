#pragma once
// SPSC ring buffers and the state shared between the callback and the inference thread.

#include "common.hpp"

// ───────────────────────── RING BUFFER ─────────────────────────
// Single-producer / single-consumer. r and w are MONOTONIC sample counters (only the buffer index
// is masked), so available() is exact and a completely full ring is distinguishable from an empty
// one — with masked indices a ring filled to capacity read back as EMPTY (w == r) and the next
// push overwrote unread data. push() is producer-only, pop()/drop() consumer-only.
struct RingBuffer {
  std::vector<float> buf;
  size_t mask;
  std::atomic<size_t> r{0}, w{0};
  explicit RingBuffer(size_t cap): buf(cap), mask(cap-1){
    assert(cap > 0 && (cap & mask)==0); // must be power-of-two
  }
  size_t capacity() const { return mask + 1; }
  size_t available() const {
    return w.load(std::memory_order_acquire) - r.load(std::memory_order_acquire);
  }
  size_t freeSpace() const {
    return capacity() - available();
  }
  size_t push(const float *data, size_t n){
    const size_t w0 = w.load(std::memory_order_relaxed);
    const size_t free = capacity() - (w0 - r.load(std::memory_order_acquire));
    const size_t toW = std::min(n, free);
    for(size_t i=0;i<toW;i++) buf[(w0+i)&mask] = data[i];
    w.store(w0+toW, std::memory_order_release);
    return toW;
  }
  size_t pop(float *out, size_t n){
    const size_t r0 = r.load(std::memory_order_relaxed);
    const size_t toR = std::min(n, w.load(std::memory_order_acquire) - r0);
    for(size_t i=0;i<toR;i++) out[i] = buf[(r0+i)&mask];
    r.store(r0+toR, std::memory_order_release);
    return toR;
  }
  // Consumer-side discard of up to n samples (no copy). Returns the number dropped.
  size_t drop(size_t n){
    const size_t r0 = r.load(std::memory_order_relaxed);
    const size_t toR = std::min(n, w.load(std::memory_order_acquire) - r0);
    r.store(r0+toR, std::memory_order_release);
    return toR;
  }
};

// Drop the same number of samples from the processed and DI rings. The inference thread pushes
// dry FIRST, then out, so dry.available() >= out.available() at any instant and dropping
// out.available() from both keeps them sample-aligned. (Flushing each ring to its own write
// pointer raced that push order: a flush between the two pushes left DI one hop ahead of the
// processed audio for good.)
static inline void dropAligned(RingBuffer& out, RingBuffer& dry, size_t n) {
  const size_t k = out.drop(n);
  dry.drop(k);
}

struct Shared {
  RingBuffer in, out, dry;
  size_t hop = 0;                 // frames per callback == model hop (fixed before the stream starts)
  size_t latencyHops = 1;         // processed hops queued ahead of the callback (1 = minimum latency)
  std::vector<float> tempIn;      // mono input (from ADC ch1)
  std::vector<float> tempOut;     // processed mono
  std::vector<float> tempStereo;  // processed stereo L/R
  std::vector<float> tempDry;     // delayed DI
  std::vector<float> tempFxReturnL;
  std::vector<float> tempFxReturnR;

  explicit Shared(size_t cap) : in(cap), out(cap), dry(cap) {}
};
