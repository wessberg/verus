/*
  stream_tflite.cpp
  Realtime PortAudio streamer using TensorFlow Lite (LiteRT) model windows.

  Updated for streaming/stateful realtime TFLite (T=32):
    - Detects realtime_state_* inputs at runtime.
    - If streaming: feeds/updates states via a packed bundle + splits outputs, and
      schedules true streaming blocks (first 'hop' samples filled, rest zeros).
    - If not streaming: falls back to original windowed schedule (left=T-hop).

  Build notes (example):
    clang++ -O3 -std=c++17 stream_tflite.cpp -o streamer_tflite \
      -ltensorflowlite -lXNNPACK -lportaudio -framework Accelerate -lpthread

  Usage example:
    ./streamer_tflite --tflite out/models/realtime/piezo2mic_baseline_realtime_t32_fp32.tflite \
                      --input-device <id> --output-device <id> --blocksize 16
*/

#include <iostream>
#include <vector>
#include <array>
#include <thread>
#include <atomic>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <csignal>
#include <cstring>
#include <cassert>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>
#include <cmath>
#include <sys/stat.h>
#include <unordered_map>
#include <string>
#include <algorithm>

#include <pthread/qos.h>
#include <Accelerate/Accelerate.h>
#include <portaudio.h>
#ifdef __APPLE__
  #include <sched.h>
  #include "pa_mac_core.h"
  #include <CoreAudio/CoreAudio.h>
  #include <CoreFoundation/CoreFoundation.h>
#endif

// TensorFlow Lite
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#ifdef TFLITE_WITH_XNNPACK
  #include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#endif

// ───────────────────────── GLOBALS/FLAGS ─────────────────────────
// Rolling window of style prob (weighted by coreLen)
const double windowSec = 2.0;

std::atomic<bool> gStopFlag(false), gBypass(false);
void signalHandler(int){ gStopFlag = true; }

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

// Round up to next power of two
static inline size_t nextPow2(size_t x) { size_t p=1; while (p<x) p<<=1; return p; }

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

// UI knobs (atomic so audio thread reads lock-free)
struct Knobs {
  std::atomic<float> pickAttack_dB{0.0f};
  std::atomic<float> pickReso_dB{0.0f};
  std::atomic<float> ir_dB{0.0f};
  std::atomic<float> wet_pct{100.0f}; // 0..100
} gKnobs;

// For latency logging
static double gIOInputLatencySec  = 0.0;
static double gIOOutputLatencySec = 0.0;
static int    PROCESSING_LOGGING_WAIT_SEC = 2;

// ───────────────────────── RING BUFFER ─────────────────────────
struct RingBuffer {
  std::vector<float> buf;
  size_t mask;
  std::atomic<size_t> r{0}, w{0};
  explicit RingBuffer(size_t cap): buf(cap), mask(cap-1){
    assert((cap & mask)==0); // must be power-of-two
  }
  size_t available() const {
    return (w.load(std::memory_order_acquire)
          - r.load(std::memory_order_acquire)) & mask;
  }
  size_t freeSpace() const {
    return (mask+1) - available();
  }
  size_t push(const float *data, size_t n){
    size_t free = freeSpace(), toW = std::min(n, free);
    size_t w0 = w.load(std::memory_order_relaxed);
    for(size_t i=0;i<toW;i++) buf[(w0+i)&mask] = data[i];
    w.store((w0+toW)&mask, std::memory_order_release);
    return toW;
  }
  size_t pop(float *out, size_t n){
    size_t av = available(), toR = std::min(n,av);
    size_t r0 = r.load(std::memory_order_relaxed);
    for(size_t i=0;i<toR;i++) out[i] = buf[(r0+i)&mask];
    r.store((r0+toR)&mask, std::memory_order_release);
    return toR;
  }
};

struct Shared {
  RingBuffer in, out;
  std::vector<float> tempOut;
  explicit Shared(size_t cap): in(cap), out(cap){}
};

// ───────────────────────── DEVICE ENUM ─────────────────────────
struct DeviceInfo {
  int devId;
  std::string name;
  int maxInputChannels;
  int maxOutputChannels;
  double defaultSampleRate;
};

static std::vector<DeviceInfo> getDevices() {
  PaError err = Pa_Initialize();
  if (err != paNoError) {
    std::cerr << "Pa_Initialize failed: " << Pa_GetErrorText(err) << "\n";
    std::exit(1);
  }
  int n = Pa_GetDeviceCount();
  if (n < 0) {
    std::cerr << "Pa_GetDeviceCount failed: " << Pa_GetErrorText(n) << "\n";
    Pa_Terminate(); std::exit(1);
  }
  std::vector<DeviceInfo> devices;
  devices.reserve(n);
  for (int i = 0; i < n; ++i) {
    const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
    if (!info) continue;
    DeviceInfo d;
    d.devId = i;
    d.name = info->name ? info->name : "";
    d.maxInputChannels  = info->maxInputChannels;
    d.maxOutputChannels = info->maxOutputChannels;
    d.defaultSampleRate = info->defaultSampleRate;
    devices.push_back(d);
  }
  Pa_Terminate();
  return devices;
}

static void listDevices(const std::vector<DeviceInfo> &devices) {
  std::cout<<"PortAudio devices:\n";
  for (const auto &d : devices) {
    std::cout<<"  "<<d.devId<<": "<<d.name
             <<" (inputs: "<<d.maxInputChannels
             <<", outputs: "<<d.maxOutputChannels
             <<", sampleRate: "<<d.defaultSampleRate<<")\n";
  }
}

// ────────────────────── NAME/DTYPE HELPERS ─────────────────────
static inline std::string normName(const char* n) {
  std::string s = n ? std::string(n) : std::string();
  if (s.rfind("serving_default_", 0) == 0)
    s = s.substr(std::strlen("serving_default_"));
  auto pos = s.find(':');
  if (pos != std::string::npos) s = s.substr(0, pos);
  // Remove TF scope prefixes like "StatefulPartitionedCall/..."
  auto slash = s.find_last_of('/');
  if (slash != std::string::npos) s = s.substr(slash+1);
  return s;
}
static inline bool isIntType(TfLiteType t) {
  return (t == kTfLiteInt32 || t == kTfLiteInt64 || t == kTfLiteUInt8 || t == kTfLiteInt16);
}

// ───────────────────────── TFLITE WRAPPER ─────────────────────────
class TFLiteRealtime {
public:
  float getStyleEma() const {
    int idx = findStateSlotIndex("realtime_state_style_ema");
    if (idx < 0) return std::numeric_limits<float>::quiet_NaN();
    const StateSlot& s = stateSlots_[idx];
    if (s.buf.empty()) return std::numeric_limits<float>::quiet_NaN();
    // state is [1,1,1] flattened
    return s.buf[0];
  }

private:
  int findStateSlotIndex(const std::string& name) const {
    for (size_t i = 0; i < stateSlots_.size(); ++i) {
      if (stateSlots_[i].name == name) return static_cast<int>(i);
    }
    return -1;
  }
public:
  explicit TFLiteRealtime(const std::string& modelPath) {
    model_ = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
    if (!model_) {
      std::cerr << "[TFLite] Cannot load model: " << modelPath << "\n";
      std::exit(1);
    }
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model_, resolver)(&interp_);
    if (!interp_) {
      std::cerr << "[TFLite] InterpreterBuilder failed\n";
      std::exit(1);
    }
    interp_->SetNumThreads(1);

    #ifdef TFLITE_WITH_XNNPACK
      {
        TfLiteXNNPackDelegateOptions opt = TfLiteXNNPackDelegateOptionsDefault();
        opt.num_threads = 1; // keep real-time friendly
        xnnpack_ = TfLiteXNNPackDelegateCreate(&opt);
        if (xnnpack_) interp_->ModifyGraphWithDelegate(xnnpack_);
      }
    #endif

    if (interp_->AllocateTensors() != kTfLiteOk) {
      std::cerr << "[TFLite] AllocateTensors failed\n";
      std::exit(1);
    }

    // Build input/output maps
    for (int i : interp_->inputs()) {
      auto* t = interp_->tensor(i);
      in_map_[normName(t->name)] = i;
    }
    for (int i : interp_->outputs()) {
      auto* t = interp_->tensor(i);
      out_map_[normName(t->name)] = i;
    }

    // Required inputs
    in_piezo_   = findInput("realtime_piezo_input");
    in_st_true_ = findInput("realtime_style_true_input");
    in_st_force_= findInput("realtime_style_force_input");
    in_knob_pa_ = findInput("realtime_knob_pickattack");
    in_knob_pr_ = findInput("realtime_knob_pickreso");
    in_knob_ir_ = findInput("realtime_knob_ir");
    in_knob_dw_ = findInput("realtime_knob_drywet");

    // Determine T
    auto* tin = (in_piezo_ >= 0) ? interp_->tensor(in_piezo_) : nullptr;
    if (!tin || tin->dims->size < 2) {
      std::cerr<<"[TFLite] Unexpected/no piezo input; want [1,T,1]\n";
      std::exit(1);
    }
    if (tin->dims->size == 3 && (tin->dims->data[0] != 1 || tin->dims->data[2] != 1)) {
      std::cerr<<"[TFLite] Unexpected piezo shape; want [1,T,1]\n";
      std::exit(1);
    }
    T_ = tin->dims->data[1];

    // Discover streaming states by input names
    discoverStateInputs();

    // Pick outputs by shape/dtype
    classifyOutputsByShape();

    // Basic checks for required non-state I/O
    if (in_piezo_ < 0 || out_audio_ < 0 || in_st_true_ < 0 || in_st_force_ < 0 ||
        in_knob_pa_ < 0 || in_knob_pr_ < 0 || in_knob_ir_ < 0 || in_knob_dw_ < 0) {
      std::cerr << "[TFLite] Missing required I/O tensors (check model names)\n";
      std::exit(1);
    }

    std::cout<<"[TFLite] Ready. T="<<T_<<"  streaming="<<(isStreaming()?"true":"false")
             <<"  states="<<stateSlots_.size()<<"\n";
  }

  ~TFLiteRealtime() {
  #ifdef TFLITE_WITH_XNNPACK
      if (xnnpack_) TfLiteXNNPackDelegateDelete(xnnpack_);
  #endif
  }

  int T() const { return T_; }
  bool isStreaming() const { return !stateSlots_.empty(); }

  // Reset state buffers (zeros); call before starting stream
  void resetStates() {
    for (auto &s : stateSlots_) {
      std::fill(s.buf.begin(), s.buf.end(), 0.0f);
    }
  }

  // STREAMING invoke:
  //   - coreLen samples are at the start of 'inBlock' (size >= coreLen),
  //     we fill the first coreLen of [1,T,1] and zero the remainder.
  //   - Returns the full T-length output block; caller will use first coreLen samples.
  void invokeStreamingBlock(const float* inBlockCore, int coreLen,
                            float* outBlockT, float &styleProb,
                            float pa_lin, float pr_lin, float ir_lin, float dry_frac,
                            float style_true, float style_force)
  {
    // Prepare piezo [1,T,1]
    float* pz = interp_->typed_tensor<float>(in_piezo_);
    std::fill(pz, pz + T_, 0.0f);
    if (coreLen > 0) std::memcpy(pz, inBlockCore, sizeof(float) * std::min(coreLen, T_));

    // Scalars [1,1]
    *interp_->typed_tensor<float>(in_st_true_)  = style_true;
    *interp_->typed_tensor<float>(in_st_force_) = style_force;

    *interp_->typed_tensor<float>(in_knob_pa_) = pa_lin;
    *interp_->typed_tensor<float>(in_knob_pr_) = pr_lin;
    *interp_->typed_tensor<float>(in_knob_ir_) = ir_lin;
    *interp_->typed_tensor<float>(in_knob_dw_) = dry_frac;

    // Feed state inputs (if streaming)
    if (isStreaming()) {
      for (auto &s : stateSlots_) {
        float* dst = interp_->typed_tensor<float>(s.input_idx);
        // Input shape is [1, L, C]; buf holds L*C contiguous floats
        std::memcpy(dst, s.buf.data(), sizeof(float) * s.len());
      }
    }

    if (interp_->Invoke() != kTfLiteOk) {
      std::cerr << "[TFLite] Invoke failed\n";
      std::memset(outBlockT, 0, sizeof(float) * T_);
      styleProb = 0.0f;
      return;
    }

    // Audio
    const float* y = interp_->typed_tensor<float>(out_audio_);
    std::memcpy(outBlockT, y, sizeof(float) * T_);

    // Style prob (2D [1,1] preferred)
    styleProb = 0.0f;
    if (out_style_ >= 0) {
      auto* st = interp_->tensor(out_style_);
      if (st && st->type == kTfLiteFloat32) {
        if (st->dims->size == 2 && st->dims->data[0] == 1 && st->dims->data[1] == 1) {
          styleProb = *interp_->typed_tensor<float>(out_style_);
        } else {
          styleProb = interp_->typed_tensor<float>(out_style_)[0];
        }
      }
    }

    // Update states from packed bundle (if present)
    if (isStreaming() && out_bundle_ >= 0) {
      const TfLiteTensor* bt = interp_->tensor(out_bundle_);
      const float* bundle = interp_->typed_tensor<float>(out_bundle_);
      int total = 1;
      for (int i = 0; i < bt->dims->size; ++i) total *= bt->dims->data[i];
      if (out_splits_ >= 0) {
        // With splits vector
        const TfLiteTensor* sp = interp_->tensor(out_splits_);
        std::vector<int> lens;
        if (sp->type == kTfLiteInt32) {
          const int32_t* p = interp_->typed_tensor<int32_t>(out_splits_);
          int N = 1;
          for (int i=0;i<sp->dims->size;i++) N *= sp->dims->data[i];
          for (int i=0;i<N;i++) lens.push_back((int)p[i]);
        } else if (sp->type == kTfLiteInt64) {
          const int64_t* p = interp_->typed_tensor<int64_t>(out_splits_);
          int N = 1;
          for (int i=0;i<sp->dims->size;i++) N *= sp->dims->data[i];
          for (int i=0;i<N;i++) lens.push_back((int)p[i]);
        } else {
          lens.clear();
        }
        // Normalize splits to 1xN
        if (sp->dims->size == 2 && sp->dims->data[0] == 1) {
          // ok
        } else if (sp->dims->size == 1) {
          // already flattened
        }
        // Update in lexicographic order (same order we built stateSlots_)
        size_t offset = 0;
        for (size_t i = 0; i < stateSlots_.size(); ++i) {
          int need = (i < lens.size()) ? lens[i] : stateSlots_[i].len();
          int have = std::max(0, std::min(need, total - (int)offset));
          if (have > 0) {
            std::memcpy(stateSlots_[i].buf.data(), bundle + offset, sizeof(float) * std::min(have, stateSlots_[i].len()));
          }
          offset += have;
        }
      } else {
        // No splits: slice by expected L*C in lexicographic order
        size_t offset = 0;
        for (auto &s : stateSlots_) {
          int have = std::max(0, std::min(s.len(), total - (int)offset));
          if (have > 0) {
            std::memcpy(s.buf.data(), bundle + offset, sizeof(float) * have);
          }
          offset += have;
        }
      }
    }
  }

  // WINDOWED invoke (legacy non-streaming model): pass full [1,T,1] window.
  void invokeWindowed(const float* inWindowT, float* outWindowT, float &styleProb,
                      float pa_lin, float pr_lin, float ir_lin, float dry_frac,
                      float style_true, float style_force)
  {
    float* pz = interp_->typed_tensor<float>(in_piezo_);
    std::memcpy(pz, inWindowT, sizeof(float) * T_);

    *interp_->typed_tensor<float>(in_st_true_)  = style_true;
    *interp_->typed_tensor<float>(in_st_force_) = style_force;
    *interp_->typed_tensor<float>(in_knob_pa_)  = pa_lin;
    *interp_->typed_tensor<float>(in_knob_pr_)  = pr_lin;
    *interp_->typed_tensor<float>(in_knob_ir_)  = ir_lin;
    *interp_->typed_tensor<float>(in_knob_dw_)  = dry_frac;

    if (interp_->Invoke() != kTfLiteOk) {
      std::cerr << "[TFLite] Invoke failed\n";
      std::memset(outWindowT, 0, sizeof(float) * T_);
      styleProb = 0.0f;
      return;
    }
    const float* y = interp_->typed_tensor<float>(out_audio_);
    std::memcpy(outWindowT, y, sizeof(float) * T_);

    styleProb = 0.0f;
    if (out_style_ >= 0) {
      auto* st = interp_->tensor(out_style_);
      if (st && st->type == kTfLiteFloat32) {
        if (st->dims->size == 2 && st->dims->data[0] == 1 && st->dims->data[1] == 1) {
          styleProb = *interp_->typed_tensor<float>(out_style_);
        } else {
          styleProb = interp_->typed_tensor<float>(out_style_)[0];
        }
      }
    }
  }

private:
  struct StateSlot {
    std::string name;
    int input_idx = -1;
    int L = 0;
    int C = 1;
    std::vector<float> buf;
    int len() const { return L * C; }
  };

  int findInput(const std::string& name) const {
    auto it = in_map_.find(name);
    if (it != in_map_.end()) return it->second;
    for (auto &kv : in_map_) {
      if (kv.first == name) return kv.second;
    }
    return -1;
  }

  void discoverStateInputs() {
    // Collect realtime_state_* inputs and sort them lexicographically by name
    std::vector<StateSlot> tmp;
    for (auto &kv : in_map_) {
      const std::string& nm = kv.first;
      if (nm.rfind("realtime_state_", 0) == 0) {
        int idx = kv.second;
        const TfLiteTensor* t = interp_->tensor(idx);
        if (!t || t->dims->size < 2) continue;
        StateSlot s;
        s.name = nm;
        s.input_idx = idx;
        // Expect shape [1, L, C] or [1, L]
        if (t->dims->size == 3) {
          s.L = t->dims->data[1];
          s.C = t->dims->data[2];
        } else if (t->dims->size == 2) {
          s.L = t->dims->data[1];
          s.C = 1;
        } else {
          continue;
        }
        s.buf.resize(std::max(1, s.L*s.C), 0.0f);
        tmp.push_back(std::move(s));
      }
    }
    std::sort(tmp.begin(), tmp.end(), [](const StateSlot& a, const StateSlot& b){
      return a.name < b.name;
    });
    stateSlots_.swap(tmp);
  }

  void classifyOutputsByShape() {
  out_audio_  = -1;
  out_style_  = -1;
  out_bundle_ = -1;
  out_splits_ = -1;

  size_t bestAudioTotal  = 0;
  size_t bestBundleTotal = 0;

  // 1) Audio = 3D float (prefer [1,T,1]); pick the largest 3D if multiple
  for (int idx : interp_->outputs()) {
    TfLiteTensor* t = interp_->tensor(idx);
    if (!t || !t->dims) continue;
    if (t->type == kTfLiteFloat32 && t->dims->size == 3) {
      size_t total = 1;
      for (int i = 0; i < t->dims->size; ++i) total *= t->dims->data[i];
      if (total > bestAudioTotal) {
        bestAudioTotal = total;
        out_audio_ = idx;
      }
    }
  }

  // 2) Style = 2D float (1,1) if present
  for (int idx : interp_->outputs()) {
    TfLiteTensor* t = interp_->tensor(idx);
    if (!t || !t->dims) continue;
    if (t->type == kTfLiteFloat32 && t->dims->size == 2 &&
        t->dims->data[0] == 1 && t->dims->data[1] == 1) {
      out_style_ = idx;
      break;
    }
  }

  // 3) Bundle = largest 2D float (excluding (1,1))
  for (int idx : interp_->outputs()) {
    TfLiteTensor* t = interp_->tensor(idx);
    if (!t || !t->dims) continue;
    if (t->type == kTfLiteFloat32 && t->dims->size == 2) {
      int r = t->dims->data[0], c = t->dims->data[1];
      if (!(r == 1 && c == 1)) {
        size_t total = static_cast<size_t>(r) * static_cast<size_t>(c);
        if (total > bestBundleTotal) {
          bestBundleTotal = total;
          out_bundle_ = idx;
        }
      }
    }
  }

  // 4) Splits = int vector (1,Nstates) or (Nstates,)
  const int nStates = static_cast<int>(stateSlots_.size());
  for (int idx : interp_->outputs()) {
    TfLiteTensor* t = interp_->tensor(idx);
    if (!t || !t->dims) continue;
    if (!isIntType(t->type)) continue;
    int dims = t->dims->size;
    if (dims == 2 && t->dims->data[0] == 1 && t->dims->data[1] == nStates) {
      out_splits_ = idx;
      break;
    }
    if (dims == 1 && t->dims->data[0] == nStates) {
      out_splits_ = idx;
      break;
    }
  }

  // Fallbacks if something still missing:
  if (out_audio_ < 0) {
    // pick any 3D float
    for (int idx : interp_->outputs()) {
      TfLiteTensor* t = interp_->tensor(idx);
      if (t && t->type == kTfLiteFloat32 && t->dims && t->dims->size == 3) {
        out_audio_ = idx;
        break;
      }
    }
    if (out_audio_ < 0) {
      // as a last resort, choose the largest tensor
      size_t best = 0;
      for (int idx : interp_->outputs()) {
        TfLiteTensor* t = interp_->tensor(idx);
        if (!t || !t->dims) continue;
        size_t total = 1;
        for (int i = 0; i < t->dims->size; ++i) total *= t->dims->data[i];
        if (total > best) { best = total; out_audio_ = idx; }
      }
    }
  }
  if (out_style_ < 0) {
    // choose any small 2D/1D float tensor
    for (int idx : interp_->outputs()) {
      TfLiteTensor* t = interp_->tensor(idx);
      if (t && t->type == kTfLiteFloat32 && t->dims && t->dims->size <= 2) {
        out_style_ = idx;
        break;
      }
    }
  }
}

  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter>     interp_;
  #ifdef TFLITE_WITH_XNNPACK
    TfLiteDelegate* xnnpack_ = nullptr;
  #else
    void* xnnpack_ = nullptr;
  #endif

  std::unordered_map<std::string, int> in_map_, out_map_;
  int in_piezo_=-1, in_st_true_=-1, in_st_force_=-1, in_knob_pa_=-1, in_knob_pr_=-1, in_knob_ir_=-1, in_knob_dw_=-1;
  int out_audio_=-1, out_style_=-1, out_bundle_=-1, out_splits_=-1;
  int T_ = 0;

  std::vector<StateSlot> stateSlots_;
};

// ───────────────────────── UTIL ─────────────────────────
static inline float db2lin(float dB) {
  return std::pow(10.0f, dB / 20.0f);
}
static inline float clampf(float x, float lo, float hi) {
  return std::max(lo, std::min(hi, x));
}

// ───────────────────────── INFERENCE THREAD ─────────────────────────
struct InferCtx {
  Shared* sh;
  int hop;
  int sampleRate;
  std::string modelPath;
};

void inferenceThread(InferCtx ctx) {
#if defined(__APPLE__)
  #ifdef QOS_CLASS_AUDIO
    pthread_set_qos_class_self_np(QOS_CLASS_AUDIO, 0);
  #else
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  #endif
  struct sched_param sp;
  sp.sched_priority = sched_get_priority_max(SCHED_FIFO);
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
#endif

  TFLiteRealtime rt(ctx.modelPath);
  const int T   = rt.T();
  const int hop = ctx.hop;

  if (hop > T) {
    std::cerr << "[inference] Error: blocksize/hop ("<<hop<<") must be <= T ("<<T<<")\n";
    gStopFlag = true; return;
  }

  // Make sure we run with hop == T for best behavior (classifier looks over full T)
  if (hop != T) {
    std::cerr << "[inference] WARN: hop ("<<hop<<") != T ("<<T<<"). Consider --blocksize "<<T<<" for best style stability.\n";
  }

  std::vector<float> xhop(hop, 0.0f);
  std::vector<float> yT(T, 0.0f);
  float styleP = 0.0f;  // raw per-hop style tensor

  size_t hopsInWindow = std::max<size_t>(1, (size_t)std::ceil(windowSec * double(ctx.sampleRate) / double(hop)));
  ProbWindow probWin;
  probWin.init(nextPow2(hopsInWindow + 8));

  // Stabilized display probability = blend(avg_5s, current_ema)
  const float kStabBlend = 0.7f;  // larger → more weight to long window
  float stable_for_next  = 0.5f;  // used if guided-auto feeds model (one-hop lag)

  // Track last printed knob values
  float prev_pa  = gKnobs.pickAttack_dB.load();
  float prev_pr  = gKnobs.pickReso_dB.load();
  float prev_ir  = gKnobs.ir_dB.load();
  float prev_wet = gKnobs.wet_pct.load();
  int   prev_mode= gStyleMode.load();

  auto pickOrFinger = [&](float p, int styleMode)->const char* {
    if (styleMode == static_cast<int>(StyleMode::ForcePick))   return "PICK (forced)";
    if (styleMode == static_cast<int>(StyleMode::ForceFinger)) return "FINGER (forced)";
    return (p >= 0.60f) ? "PICK" : "FINGER"; // slightly conservative UI threshold
  };

  auto printKnobStatus = [&](const char* tag, float dispP){
    int sm = gStyleMode.load();
    std::cout << (tag ? tag : "[UI]") << " PA=" << gKnobs.pickAttack_dB.load() << " dB  "
              << "PR=" << gKnobs.pickReso_dB.load()   << " dB  "
              << "IR=" << gKnobs.ir_dB.load()         << " dB  "
              << "Wet="<< gKnobs.wet_pct.load()       << " %  "
              << "Pred P=" << dispP << "  "
              << "Pred=" << pickOrFinger(dispP, sm) << "  "
              << "Mode=" << styleModeName(sm)
              << (gGuidedAuto.load() ? "  (Guided AUTO ON)" : "")
              << "\n";
  };

  auto checkAndReportKnobChanges = [&](float dispP){
    float pa  = gKnobs.pickAttack_dB.load();
    float pr  = gKnobs.pickReso_dB.load();
    float ir  = gKnobs.ir_dB.load();
    float wet = gKnobs.wet_pct.load();
    int   sm  = gStyleMode.load();
    if (pa != prev_pa || pr != prev_pr || ir != prev_ir || wet != prev_wet || sm != prev_mode) {
      printKnobStatus("[UI]", dispP);
      prev_pa = pa; prev_pr = pr; prev_ir = ir; prev_wet = wet; prev_mode = sm;
    }
  };

  // Wait for at least one hop of input
  while (!gStopFlag && ctx.sh->in.available() < (size_t)hop)
    std::this_thread::yield();
  if (gStopFlag) return;

  const bool streaming = rt.isStreaming();
  if (streaming) {
    rt.resetStates();
  }

  // Initial hop
  size_t got = ctx.sh->in.pop(xhop.data(), hop);
  if (got < (size_t)hop) std::fill(xhop.begin()+got, xhop.end(), 0.0f);

  // Helpers to call TFLite for one block
  auto do_invoke_streaming = [&](const float* coreBuf, int coreLen, float guidedStableForAuto){
    float pa_lin = std::pow(10.0f, gKnobs.pickAttack_dB.load() / 20.0f);
    float pr_lin = std::pow(10.0f, gKnobs.pickReso_dB.load()   / 20.0f);
    float ir_lin = std::pow(10.0f, gKnobs.ir_dB.load()         / 20.0f);
    float wet    = std::max(0.0f, std::min(100.0f, gKnobs.wet_pct.load()));
    float dryFrac= 1.0f - (wet / 100.0f);

    int sm = gStyleMode.load();
    float st_true = 0.0f, st_force = 0.0f;
    if (sm == static_cast<int>(StyleMode::ForcePick))   { st_true = 1.0f; st_force = 1.0f; }
    if (sm == static_cast<int>(StyleMode::ForceFinger)) { st_true = 0.0f; st_force = 1.0f; }
    if (sm == static_cast<int>(StyleMode::Auto) && gGuidedAuto.load()) {
      st_true  = std::max(0.0f, std::min(1.0f, guidedStableForAuto));
      st_force = 1.0f; // override model’s internal style with 5s-stabilized prob
    }

    rt.invokeStreamingBlock(coreBuf, coreLen, yT.data(), styleP, pa_lin, pr_lin, ir_lin, dryFrac, st_true, st_force);
  };

  auto do_invoke_windowed = [&](const float* windowT, float guidedStableForAuto){
    float pa_lin = std::pow(10.0f, gKnobs.pickAttack_dB.load() / 20.0f);
    float pr_lin = std::pow(10.0f, gKnobs.pickReso_dB.load()   / 20.0f);
    float ir_lin = std::pow(10.0f, gKnobs.ir_dB.load()         / 20.0f);
    float wet    = std::max(0.0f, std::min(100.0f, gKnobs.wet_pct.load()));
    float dryFrac= 1.0f - (wet / 100.0f);

    int sm = gStyleMode.load();
    float st_true = 0.0f, st_force = 0.0f;
    if (sm == static_cast<int>(StyleMode::ForcePick))   { st_true = 1.0f; st_force = 1.0f; }
    if (sm == static_cast<int>(StyleMode::ForceFinger)) { st_true = 0.0f; st_force = 1.0f; }
    if (sm == static_cast<int>(StyleMode::Auto) && gGuidedAuto.load()) {
      st_true  = std::max(0.0f, std::min(1.0f, guidedStableForAuto));
      st_force = 1.0f;
    }

    rt.invokeWindowed(windowT, yT.data(), styleP, pa_lin, pr_lin, ir_lin, dryFrac, st_true, st_force);
  };

  // First block
  if (streaming) {
    do_invoke_streaming(xhop.data(), hop, stable_for_next);
    ctx.sh->out.push(yT.data(), hop);
    // Update 5s window with EMA (or raw) from this hop
    float emaP = rt.getStyleEma();
    float pHop = std::isfinite(emaP) ? emaP : styleP;
    probWin.push(pHop, float(hop));
    float pAvgWindowSec = probWin.mean();
    float dispP = (1.0f - kStabBlend) * pHop + kStabBlend * pAvgWindowSec;
    stable_for_next = dispP;
    checkAndReportKnobChanges(dispP);
  } else {
    // Legacy windowed: left = T - hop, right=0
    const int left = T - hop;
    std::vector<float> win(T, 0.0f);
    std::memcpy(win.data() + left, xhop.data(), sizeof(float)*hop);
    do_invoke_windowed(win.data(), stable_for_next);
    ctx.sh->out.push(yT.data() + left, hop);
    float emaP = rt.getStyleEma();
    float pHop = std::isfinite(emaP) ? emaP : styleP;
    probWin.push(pHop, float(hop));
    float pAvgWindowSec = probWin.mean();
    float dispP = (1.0f - kStabBlend) * pHop + kStabBlend * pAvgWindowSec;
    stable_for_next = dispP;
    checkAndReportKnobChanges(dispP);
  }

  // Main loop
  if (streaming) {
    while (!gStopFlag) {
      while (!gStopFlag && ctx.sh->in.available() < (size_t)hop)
        std::this_thread::yield();
      if (gStopFlag) break;

      size_t got2 = ctx.sh->in.pop(xhop.data(), hop);
      if (got2 < (size_t)hop) std::fill(xhop.begin()+got2, xhop.end(), 0.0f);

      auto t0 = std::chrono::steady_clock::now();

      // Use the previous stabilized prob for guided AUTO (one-hop lag)
      do_invoke_streaming(xhop.data(), hop, stable_for_next);
      ctx.sh->out.push(yT.data(), hop);

      // Update rolling {windowSec}s window and stabilized display prob
      float emaP = rt.getStyleEma();
      float pHop = std::isfinite(emaP) ? emaP : styleP;
      probWin.push(pHop, float(hop));
      float pAvgWindowSec = probWin.mean();
      float dispP = (1.0f - kStabBlend) * pHop + kStabBlend * pAvgWindowSec;
      stable_for_next = dispP;

      auto t1 = std::chrono::steady_clock::now();
      checkAndReportKnobChanges(dispP);

      double proc_ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
      static auto   last_report = t1;
      static double sum_proc = 0.0;
      static size_t cnt = 0;
      sum_proc += proc_ms; ++cnt;
      if (t1 - last_report >= std::chrono::seconds(PROCESSING_LOGGING_WAIT_SEC)) {
        double avg_proc = sum_proc / double(cnt);
        double io_ms    = (gIOInputLatencySec + gIOOutputLatencySec)*1000.0;
        double total_ms = avg_proc + io_ms;
        int sm = gStyleMode.load();
        std::cerr << "[inference/stream] Proc="<<avg_proc<<" ms  I/O="<<io_ms<<" ms  Total~"<<total_ms<<" ms"
                  << "  Pred" << windowSec << "s=" << pAvgWindowSec << "  DispP=" << dispP
                  << "  pred="<<pickOrFinger(dispP, sm)<<"  mode="<<styleModeName(sm)
                  << (gGuidedAuto.load() ? "  (Guided)" : "")
                  << "\n";
        last_report = t1; sum_proc = 0.0; cnt = 0;
      }
    }
  } else {
    const int left = T - hop;
    std::vector<float> win(T, 0.0f);
    while (!gStopFlag) {
      while (!gStopFlag && ctx.sh->in.available() < (size_t)hop)
        std::this_thread::yield();
      if (gStopFlag) break;

      size_t got2 = ctx.sh->in.pop(xhop.data(), hop);
      if (got2 < (size_t)hop) std::fill(xhop.begin()+got2, xhop.end(), 0.0f);

      auto t0 = std::chrono::steady_clock::now();

      // Slide window left by hop; append new hop at end
      if (left > 0) std::memmove(win.data(), win.data()+hop, sizeof(float)*(T-hop));
      std::memcpy(win.data() + (T - hop), xhop.data(), sizeof(float)*hop);

      do_invoke_windowed(win.data(), stable_for_next);
      ctx.sh->out.push(yT.data() + (T - hop), hop);

      float emaP = rt.getStyleEma();
      float pHop = std::isfinite(emaP) ? emaP : styleP;
      probWin.push(pHop, float(hop));
      float pAvgWindowSec = probWin.mean();
      float dispP = (1.0f - kStabBlend) * pHop + kStabBlend * pAvgWindowSec;
      stable_for_next = dispP;

      auto t1 = std::chrono::steady_clock::now();
      checkAndReportKnobChanges(dispP);

      double proc_ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
      static auto   last_report = t1;
      static double sum_proc = 0.0;
      static size_t cnt = 0;
      sum_proc += proc_ms; ++cnt;
      if (t1 - last_report >= std::chrono::seconds(PROCESSING_LOGGING_WAIT_SEC)) {
        double avg_proc = sum_proc / double(cnt);
        double io_ms    = (gIOInputLatencySec + gIOOutputLatencySec)*1000.0;
        double total_ms = avg_proc + io_ms;
        int sm = gStyleMode.load();
        std::cerr << "[inference/window] Proc="<<avg_proc<<" ms  I/O="<<io_ms<<" ms  Total~"<<total_ms<<" ms"
                  << "  Pred" << windowSec << "s=" << pAvgWindowSec << "  DispP=" << dispP
                  << "  pred="<<pickOrFinger(dispP, sm)<<"  mode="<<styleModeName(sm)
                  << (gGuidedAuto.load() ? "  (Guided)" : "")
                  << "\n";
        last_report = t1; sum_proc = 0.0; cnt = 0;
      }
    }
  }
}

// ───────────────────────── PA CALLBACK ─────────────────────────
static int paCallback(const void *inputBuffer,
                      void *outputBuffer,
                      unsigned long framesPerBuffer,
                      const PaStreamCallbackTimeInfo *,
                      PaStreamCallbackFlags,
                      void *userData)
{
  auto *sh = static_cast<Shared*>(userData);
  auto *in = static_cast<const float*>(inputBuffer);
  auto *out= static_cast<float*>(outputBuffer);
  size_t hop = framesPerBuffer;

  if (in) sh->in.push(in, hop);

  if (gBypass.load()) {
    if (in) {
      for (size_t i=0;i<hop;i++){
        float v = in[i];
        out[2*i] = v; out[2*i+1] = v;
      }
    } else {
      std::memset(out, 0, sizeof(float)*2*hop);
    }
    return paContinue;
  }

  float *temp = sh->tempOut.data();
  size_t got  = sh->out.pop(temp, hop);
  if (got < hop) std::memset(temp+got, 0, sizeof(float)*(hop-got));
  for (size_t i=0;i<hop;i++){
    float v = temp[i];
    out[2*i] = v; out[2*i+1] = v;
  }
  return paContinue;
}

// ───────────────────────── MAC HAL BUFFER SIZE ─────────────────────────
#ifdef __APPLE__
static void setMacHardwareBufferSize(const DeviceInfo& paDev, int blockSize, bool isInput) {
  AudioObjectID coreAudioDeviceID = kAudioObjectUnknown;

  // Enumerate devices to match by name
  AudioObjectPropertyAddress propAddr = {
    kAudioHardwarePropertyDevices,
    kAudioObjectPropertyScopeGlobal,
    kAudioObjectPropertyElementMain
  };
  UInt32 dataSize = 0;
  OSStatus deviceErr = AudioObjectGetPropertyDataSize(
      kAudioObjectSystemObject, &propAddr, 0, nullptr, &dataSize);
  if (deviceErr) { std::cerr << "Error get dev size: " << deviceErr << "\n"; return; }

  int deviceCount = dataSize / sizeof(AudioDeviceID);
  std::vector<AudioDeviceID> devs(deviceCount);
  deviceErr = AudioObjectGetPropertyData(
      kAudioObjectSystemObject, &propAddr, 0, nullptr, &dataSize, devs.data());
  if (deviceErr) { std::cerr << "Error get dev list: " << deviceErr << "\n"; return; }

  for (auto devID : devs) {
    AudioObjectPropertyAddress nameAddr = {
      kAudioObjectPropertyName,
      kAudioObjectPropertyScopeGlobal,
      kAudioObjectPropertyElementMain
    };
    CFStringRef cfName = nullptr;
    UInt32 nSize = sizeof(cfName);
    OSStatus err = AudioObjectGetPropertyData(
        devID, &nameAddr, 0, nullptr, &nSize, &cfName);
    if (err == noErr && cfName) {
      char nameBuf[256] = {0};
      CFStringGetCString(cfName, nameBuf, sizeof(nameBuf), kCFStringEncodingUTF8);
      CFRelease(cfName);
      if (std::string(nameBuf) == paDev.name) {
        coreAudioDeviceID = devID; break;
      }
    }
  }
  if (coreAudioDeviceID == kAudioObjectUnknown) {
    std::cerr << "[HAL] could not find CoreAudio device by name: " << paDev.name << "\n";
    return;
  }

  // Set buffer frame size
  UInt32 newBufferSize = (UInt32)blockSize;
  AudioObjectPropertyAddress bufAddr = {
    kAudioDevicePropertyBufferFrameSize,
    kAudioObjectPropertyScopeGlobal,
    kAudioObjectPropertyElementMain
  };
  deviceErr = AudioObjectSetPropertyData(
      coreAudioDeviceID, &bufAddr, 0, nullptr, sizeof(newBufferSize), &newBufferSize);
  if (deviceErr) {
    std::cerr << "Error setting buffer size: " << deviceErr << "\n";
  }
  // Read back
  UInt32 actualSize = 0; UInt32 asize = sizeof(actualSize);
  deviceErr = AudioObjectGetPropertyData(coreAudioDeviceID, &bufAddr, 0, nullptr, &asize, &actualSize);
  if (!deviceErr) {
    std::cout << "[HAL] " << (isInput?"Input":"Output") << " buffer now: " << actualSize << " frames\n";
  }
}
#endif

// ───────────────────────── MAIN ─────────────────────────
int main(int argc, char** argv) {
  int  blockSize = 32; // must be <= T
  int  inDevId = -1, outDevId = -1;
  std::string tflitePath;

  bool listDev = false;
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"--list-devices")) listDev = true;
    else if(!strcmp(argv[i],"--blocksize") && i+1<argc) blockSize = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--input-device") && i+1<argc) inDevId  = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--output-device")&&i+1<argc) outDevId = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--tflite") && i+1<argc) tflitePath = argv[++i];
    else {
      std::cerr<<"Unknown arg "<<argv[i]<<"\n";
      return 1;
    }
  }

  auto devices = getDevices();
  if (listDev) { listDevices(devices); return 0; }

  if (inDevId<0 || outDevId<0 || tflitePath.empty()) {
    std::cerr << "Usage: --tflite <model.tflite> --input-device <id> --output-device <id> [--blocksize N]\n";
    listDevices(devices);
    return 1;
  }

  DeviceInfo inDev{}, outDev{};
  for (auto &d : devices) {
    if (d.devId == inDevId)  inDev = d;
    if (d.devId == outDevId) outDev = d;
  }
  if (inDev.devId<0 || outDev.devId<0) {
    std::cerr<<"Invalid device IDs\n";
    listDevices(devices);
    return 1;
  }

#ifdef __APPLE__
  setMacHardwareBufferSize(inDev,  blockSize, true);
  setMacHardwareBufferSize(outDev, blockSize, false);
#endif

  PaError err = Pa_Initialize();
  if (err != paNoError) {
    std::cerr<<"Pa_Initialize failed: "<<Pa_GetErrorText(err)<<"\n";
    return 1;
  }
  auto *diIn  = Pa_GetDeviceInfo(inDevId);
  auto *diOut = Pa_GetDeviceInfo(outDevId);
  int sampleRate = (int)std::lround(diIn->defaultSampleRate);

  std::cout<<"[MAIN] In ["<<inDevId<<"] "<<diIn->name<<" sr="<<diIn->defaultSampleRate<<"\n";
  std::cout<<"[MAIN] Out["<<outDevId<<"] "<<diOut->name<<" sr="<<diOut->defaultSampleRate<<"\n";
  std::cout<<"[MAIN] blockSize="<<blockSize<<"\n";
  std::cout<<"[MAIN] model: "<<tflitePath<<"\n";

  // PortAudio stream params
  double wantLatencySecs = double(blockSize) / sampleRate;
  PaStreamParameters ip = { inDevId, 1, paFloat32, wantLatencySecs, nullptr };
  PaStreamParameters op = { outDevId, 2, paFloat32, wantLatencySecs, nullptr };

  // stdin non-blocking for UI
  termios oldt,newt;
  tcgetattr(STDIN_FILENO,&oldt);
  newt = oldt; newt.c_lflag &= ~(ICANON|ECHO);
  tcsetattr(STDIN_FILENO,TCSANOW,&newt);
  fcntl(STDIN_FILENO,F_SETFL,O_NONBLOCK);

  std::signal(SIGINT, signalHandler);

  // Shared rings
  size_t cap = 1<<14;
  Shared shared(cap);
  shared.tempOut.resize(blockSize);

  // Spawn inference thread
  InferCtx ictx;
  ictx.sh = &shared; ictx.hop = blockSize; ictx.sampleRate = sampleRate; ictx.modelPath = tflitePath;
  std::thread infThread([&](){ inferenceThread(ictx); });

  // Keyboard thread for bypass + knobs
  std::thread kbThread([&](){

  std::cout
    << "[MAIN] Keys: "
    << "b=Bypass  "
    << "q/a=PA +/-1dB  "
    << "w/s=PR +/-1dB  "
    << "e/d=IR +/-1dB  "
    << "r/f=Wet +/-5%  "
    << "t=Style mode (auto→pick→finger)  "
    << "g=Guided AUTO on/off  "
    << "p=print"
    << "\n";

    while (!gStopFlag) {
      int c = getchar();
      if (c == EOF) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); continue; }
      if (c=='b'||c=='B') {
        bool now = !gBypass.load(); gBypass.store(now);
        shared.out.r.store(shared.out.w.load(std::memory_order_relaxed), std::memory_order_relaxed);
        std::cout<<"[UI] Bypass "<<(now?"ON":"OFF")<<"\n";
      } else if (c=='q') { gKnobs.pickAttack_dB.store( clampf(gKnobs.pickAttack_dB.load()+1.0f, -24.0f, 24.0f) ); }
        else if (c=='a') { gKnobs.pickAttack_dB.store( clampf(gKnobs.pickAttack_dB.load()-1.0f, -24.0f, 24.0f) ); }
        else if (c=='w') { gKnobs.pickReso_dB.store( clampf(gKnobs.pickReso_dB.load()+1.0f, -24.0f, 24.0f) ); }
        else if (c=='s') { gKnobs.pickReso_dB.store( clampf(gKnobs.pickReso_dB.load()-1.0f, -24.0f, 24.0f) ); }
        else if (c=='e') { gKnobs.ir_dB.store( clampf(gKnobs.ir_dB.load()+1.0f, -24.0f, 24.0f) ); }
        else if (c=='d') { gKnobs.ir_dB.store( clampf(gKnobs.ir_dB.load()-1.0f, -24.0f, 24.0f) ); }
        else if (c=='r') { gKnobs.wet_pct.store( clampf(gKnobs.wet_pct.load()+5.0f, 0.0f, 100.0f) ); }
        else if (c=='f') { gKnobs.wet_pct.store( clampf(gKnobs.wet_pct.load()-5.0f, 0.0f, 100.0f) ); }
        else if (c=='t') {
          int cur = gStyleMode.load();
          int next = (cur == static_cast<int>(StyleMode::Auto)) ? static_cast<int>(StyleMode::ForcePick)
                     : (cur == static_cast<int>(StyleMode::ForcePick)) ? static_cast<int>(StyleMode::ForceFinger)
                     : static_cast<int>(StyleMode::Auto);
          gStyleMode.store(next);
          std::cout<<"[UI] Style mode = "<<styleModeName(next)<<"\n";
        }
        else if (c=='p') {
          std::cout<<"[UI] PA="<<gKnobs.pickAttack_dB.load()<<" dB  "
                   <<"PR="<<gKnobs.pickReso_dB.load()<<" dB  "
                   <<"IR="<<gKnobs.ir_dB.load()<<" dB  "
                   <<"Wet="<<gKnobs.wet_pct.load()<<" %  "
                   <<"Style="<<styleModeName(gStyleMode.load())<<"\n";
        }

        else if (c=='g' || c=='G') {
          bool now = !gGuidedAuto.load();
          gGuidedAuto.store(now);
          std::cout << "[UI] Guided AUTO style " << (now ? "ON (model style overridden by " + std::to_string(windowSec) + "s-stabilized prob)" : "OFF") << "\n";
        }
    }
  });

  PaStream *stream = nullptr;
  err = Pa_OpenStream(&stream, &ip, &op, sampleRate, (unsigned long)blockSize, paNoFlag, paCallback, &shared);
  if (err != paNoError) {
    std::cerr<<"Pa_OpenStream failed: "<<Pa_GetErrorText(err)<<"\n";
    Pa_Terminate(); gStopFlag = true;
    infThread.join(); kbThread.join();
    tcsetattr(STDIN_FILENO,TCSANOW,&oldt);
    return 1;
  }
  err = Pa_StartStream(stream);
  if (err != paNoError) {
    std::cerr<<"Pa_StartStream failed: "<<Pa_GetErrorText(err)<<"\n";
    Pa_CloseStream(stream); Pa_Terminate(); gStopFlag = true;
    infThread.join(); kbThread.join();
    tcsetattr(STDIN_FILENO,TCSANOW,&oldt);
    return 1;
  }

  auto *info = Pa_GetStreamInfo(stream);
  gIOInputLatencySec  = info->inputLatency;
  gIOOutputLatencySec = info->outputLatency;
  std::cerr<<"[MAIN] Actual I/O buffer: in="<<(int)std::round(gIOInputLatencySec*sampleRate)
           <<" out="<<(int)std::round(gIOOutputLatencySec*sampleRate)<<" frames\n";

  std::cout<<"[MAIN] Streaming… (Ctrl-C to quit)\n";
  while (!gStopFlag) Pa_Sleep(100);

  std::cout<<"[MAIN] Stopping…\n";
  Pa_StopStream(stream);
  Pa_CloseStream(stream);
  Pa_Terminate();

  infThread.join();
  kbThread.join();
  tcsetattr(STDIN_FILENO,TCSANOW,&oldt);
  std::cout<<"[MAIN] Done.\n";
  return 0;
}