#pragma once
// Small numeric helpers.

#include "common.hpp"

// Round up to next power of two
static inline size_t nextPow2(size_t x) { size_t p=1; while (p<x) p<<=1; return p; }

// ───────────────────────── UTIL ─────────────────────────
static inline float db2lin(float dB) {
  return std::pow(10.0f, dB / 20.0f);
}
static inline float clampf(float x, float lo, float hi) {
  return std::max(lo, std::min(hi, x));
}
