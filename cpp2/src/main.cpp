/*
  stream_tflite.cpp
  Realtime PortAudio streamer using TensorFlow Lite (LiteRT) model windows.

  - Preserves macOS HAL buffer size control (sets hardware buffer = --blocksize).
  - Inference schedule: left = T - hop, core = hop, right = 0 (like your Python runner).
  - Knobs via keys:
      b = bypass toggle
      q/a = pick attack +1/-1 dB
      w/s = pick reso   +1/-1 dB
      e/d = IR          +1/-1 dB
      r/f = wet         +5/-5 %
      p   = print knob values

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

// ────────────────────── NAME NORMALIZATION ─────────────────────
static inline std::string normName(const char* n) {
  std::string s = n ? std::string(n) : std::string();
  if (s.rfind("serving_default_", 0) == 0)
    s = s.substr(std::strlen("serving_default_"));
  auto pos = s.find(':');
  if (pos != std::string::npos) s = s.substr(0, pos);
  return s;
}

// ───────────────────────── TFLITE WRAPPER ─────────────────────────
class TFLiteRealtime {
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
        // XNNPACK delegate
        TfLiteXNNPackDelegateOptions opt = TfLiteXNNPackDelegateOptionsDefault();
        opt.num_threads = 1; // keep real-time friendly
        xnnpack_ = TfLiteXNNPackDelegateCreate(&opt);
        if (xnnpack_) interp_->ModifyGraphWithDelegate(xnnpack_);
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
    in_piezo_ = findInput("realtime_piezo_input");
    in_st_true_ = findInput("realtime_style_true_input");
    in_st_force_= findInput("realtime_style_force_input");
    in_knob_pa_ = findInput("realtime_knob_pickattack");
    in_knob_pr_ = findInput("realtime_knob_pickreso");
    in_knob_ir_ = findInput("realtime_knob_ir");
    in_knob_dw_ = findInput("realtime_knob_drywet");

    // Determine T
    auto* tin = interp_->tensor(in_piezo_);
    if (tin->dims->size != 3 || tin->dims->data[0] != 1 || tin->dims->data[2] != 1) {
      std::cerr<<"[TFLite] Unexpected piezo input shape; want [1,T,1]\n";
      std::exit(1);
    }
    T_ = tin->dims->data[1];

    // Outputs: prefer named, else pick by size
    out_audio_ = findOutputPrefer({"realtime_audio_out"});
    out_style_ = findOutputPrefer({"realtime_style_prob_out"});

    if (out_audio_ < 0) {
      // largest output is audio
      size_t best = 0;
      for (auto &kv : out_map_) {
        auto* t = interp_->tensor(kv.second);
        size_t sz = 1;
        for (int i = 0; i < t->dims->size; ++i) sz *= t->dims->data[i];
        if (sz > best) { best = sz; out_audio_ = kv.second; }
      }
    }
    if (out_style_ < 0) {
      // choose a smallest second output or any [1,1]
      for (auto &kv : out_map_) {
        auto* t = interp_->tensor(kv.second);
        if (t->dims->size == 2 && t->dims->data[0] == 1 && t->dims->data[1] == 1) {
          out_style_ = kv.second; break;
        }
      }
      if (out_style_ < 0) {
        // fallback: pick any non-audio
        for (auto &kv : out_map_) if (kv.second != out_audio_) { out_style_ = kv.second; break; }
      }
    }

    if (in_piezo_ < 0 || out_audio_ < 0 || in_st_true_ < 0 || in_st_force_ < 0 ||
        in_knob_pa_ < 0 || in_knob_pr_ < 0 || in_knob_ir_ < 0 || in_knob_dw_ < 0) {
      std::cerr << "[TFLite] Missing required I/O tensors (check model names)\n";
      std::exit(1);
    }

    std::cout<<"[TFLite] Ready. T="<<T_<<"\n";
  }

  ~TFLiteRealtime() {
  #ifdef TFLITE_WITH_XNNPACK
      if (xnnpack_) TfLiteXNNPackDelegateDelete(xnnpack_);
  #endif
    }

  int T() const { return T_; }

  // Invoke on a [T] mono window. Outputs [T] mono audio and style prob.
  void invoke(const float* inWindow, float* outWindow, float &styleProb,
              float pa_lin, float pr_lin, float ir_lin, float dry_frac,
              float style_true, float style_force) {
    // Set inputs
    float* pz = interp_->typed_tensor<float>(in_piezo_);
    for (int t = 0; t < T_; ++t) pz[t] = inWindow[t];

    // Scalars: [1,1]
    *interp_->typed_tensor<float>(in_st_true_)  = style_true;
    *interp_->typed_tensor<float>(in_st_force_) = style_force;

    *interp_->typed_tensor<float>(in_knob_pa_) = pa_lin;
    *interp_->typed_tensor<float>(in_knob_pr_) = pr_lin;
    *interp_->typed_tensor<float>(in_knob_ir_) = ir_lin;
    *interp_->typed_tensor<float>(in_knob_dw_) = dry_frac;

    if (interp_->Invoke() != kTfLiteOk) {
      std::cerr << "[TFLite] Invoke failed\n";
      std::memset(outWindow, 0, sizeof(float) * T_);
      styleProb = 0.0f;
      return;
    }

    const float* y = interp_->typed_tensor<float>(out_audio_);
    auto* st = interp_->tensor(out_style_);
    float s = 0.0f;
    if (st && st->type == kTfLiteFloat32) {
      if (st->dims->size == 2 && st->dims->data[0] == 1 && st->dims->data[1] == 1) {
        s = *interp_->typed_tensor<float>(out_style_);
      } else {
        s = interp_->typed_tensor<float>(out_style_)[0];
      }
    }
    std::memcpy(outWindow, y, sizeof(float) * T_);
    styleProb = s;
  }

private:
  int findInput(const std::string& name) const {
    auto it = in_map_.find(name);
    if (it != in_map_.end()) return it->second;
    // fallback: try with/without serving_default prefix already stripped by normName
    for (auto &kv : in_map_) {
      if (kv.first == name) return kv.second;
    }
    return -1;
  }
  int findOutputPrefer(const std::vector<std::string>& prefer) const {
    for (auto &n : prefer) {
      auto it = out_map_.find(n);
      if (it != out_map_.end()) return it->second;
    }
    return -1;
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
  int out_audio_=-1, out_style_=-1;
  int T_ = 0;
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
  // QoS boost
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
  const int T = rt.T();
  const int hop = ctx.hop;

  if (hop > T) {
    std::cerr << "[inference] Error: blocksize/hop ("<<hop<<") must be <= T ("<<T<<")\n";
    gStopFlag = true; return;
  }
  const int left = T - hop; // schedule: left, core, right=0

  std::vector<float> win(T, 0.0f), y(T, 0.0f);

  // pre-fill left context with zeros; wait enough input for first core
  while (!gStopFlag && ctx.sh->in.available() < (size_t)hop)
    std::this_thread::yield();
  if (gStopFlag) return;

  std::vector<float> xhop(hop);
  size_t got = ctx.sh->in.pop(xhop.data(), hop);
  if (got < (size_t)hop)
    std::fill(xhop.begin()+got, xhop.end(), 0.0f);

  // First window: place core at [left : left+hop)
  // left context is zeros initially
  std::memcpy(win.data() + left, xhop.data(), sizeof(float)*hop);

  float styleP = 0.0f;

  auto do_invoke = [&](const float* inBuf) {
    // Read knobs atomically
    float pa_lin = db2lin(gKnobs.pickAttack_dB.load());
    float pr_lin = db2lin(gKnobs.pickReso_dB.load());
    float ir_lin = db2lin(gKnobs.ir_dB.load());
    float wet    = clampf(gKnobs.wet_pct.load(), 0.0f, 100.0f);
    float dryFrac= 1.0f - (wet / 100.0f);

    int sm = gStyleMode.load();
    float st_true = 0.0f, st_force = 0.0f; // AUTO default
    if (sm == static_cast<int>(StyleMode::ForcePick))   { st_true = 1.0f; st_force = 1.0f; }
    if (sm == static_cast<int>(StyleMode::ForceFinger)) { st_true = 0.0f; st_force = 1.0f; }

    rt.invoke(inBuf, y.data(), styleP, pa_lin, pr_lin, ir_lin, dryFrac, st_true, st_force);
  };

  // Warm invoke
  do_invoke(win.data());
  // push core region
  ctx.sh->out.push(y.data() + left, hop);

  // steady-state
  while (!gStopFlag) {
    // Wait next hop
    while (!gStopFlag && ctx.sh->in.available() < (size_t)hop)
      std::this_thread::yield();
    if (gStopFlag) break;

    size_t got2 = ctx.sh->in.pop(xhop.data(), hop);
    if (got2 < (size_t)hop)
      std::fill(xhop.begin()+got2, xhop.end(), 0.0f);

    auto t0 = std::chrono::steady_clock::now();

    // Slide window left by hop; append new hop at end
    if (left > 0) std::memmove(win.data(), win.data()+hop, sizeof(float)*(T-hop));
    std::memcpy(win.data() + (T - hop), xhop.data(), sizeof(float)*hop);

    do_invoke(win.data());

    // Push out core region (last hop)
    ctx.sh->out.push(y.data() + (T - hop), hop);

    auto t1 = std::chrono::steady_clock::now();
    double proc_ms = std::chrono::duration<double,std::milli>(t1 - t0).count();

    static auto   last_report = t1;
    static double sum_proc = 0.0;
    static size_t cnt = 0;
    sum_proc += proc_ms; ++cnt;
    if (t1 - last_report >= std::chrono::seconds(PROCESSING_LOGGING_WAIT_SEC)) {
      double avg_proc = sum_proc / double(cnt);
      double io_ms    = (gIOInputLatencySec + gIOOutputLatencySec)*1000.0;
      double total_ms = avg_proc + io_ms;
      std::cerr << "[inference] Proc="<<avg_proc<<" ms  I/O="<<io_ms<<" ms  Total~"<<total_ms<<" ms"
                << "  style_p≈"<<styleP << "  mode="<<styleModeName(gStyleMode.load()) << "\n";
      last_report = t1; sum_proc = 0.0; cnt = 0;
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
  int  blockSize = 16; // keep small; must be <= T
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
  std::cout<<"[MAIN] Keys: b=Bypass  q/a=PA +/‑1dB  w/s=PR +/‑1dB  e/d=IR +/‑1dB  r/f=Wet +/‑5%  t=Style mode (auto→pick→finger)  p=print\n";
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