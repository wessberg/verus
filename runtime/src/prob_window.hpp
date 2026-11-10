#pragma once
// Rolling weighted mean of the style probability.

#include "common.hpp"

// Rolling weighted average over a circular buffer of fixed power-of-two capacity
struct ProbWindow {
  std::vector<float> pbuf;
  std::vector<float> wbuf;
  size_t mask = 0;
  size_t pos  = 0;
  double sumP = 0.0;   // sum of p*w
  double sumW = 0.0;   // sum of w

  void init(size_t capPow2) {
    size_t cap = capPow2 ? capPow2 : 1;
    pbuf.assign(cap, 0.5f);
    wbuf.assign(cap, 0.0f);
    mask = cap - 1;
    pos  = 0;
    sumP = 0.0;
    sumW = 0.0;
  }
  void reset() {
    std::fill(pbuf.begin(), pbuf.end(), 0.5f);
    std::fill(wbuf.begin(), wbuf.end(), 0.0f);
    pos  = 0;
    sumP = 0.0;
    sumW = 0.0;
  }
  // push one sample with weight (e.g., coreLen)
  void push(float p, float w) {
    const size_t i = pos & mask;
    // remove old
    sumP -= double(pbuf[i]) * double(wbuf[i]);
    sumW -= double(wbuf[i]);
    // insert new
    pbuf[i] = p;
    wbuf[i] = std::max(0.0f, w);
    sumP += double(pbuf[i]) * double(wbuf[i]);
    sumW += double(wbuf[i]);
    ++pos;
  }
  float mean() const {
    return (sumW > 1e-9 ? float(sumP / sumW) : 0.5f);
  }
};
