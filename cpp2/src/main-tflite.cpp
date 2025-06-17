// stream_piezo2mic.cpp
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstring>
#include <csignal>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>
#include <portaudio.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

static constexpr int FILTERS = 32;
static constexpr int KERNEL_SIZE = 63;
static constexpr int MAX_SUPPORTED_BLOCK_SIZE = 512;
static constexpr int DILATIONS[] = {1, 2, 4, 8, 16, 32}; // 7 dilations
static constexpr int RECEPTIVE_FIELD = 1 + (KERNEL_SIZE - 1) * (DILATIONS[0] + DILATIONS[1] + DILATIONS[2] + DILATIONS[3] + DILATIONS[4] + DILATIONS[5]);
static constexpr int SEG_LEN = RECEPTIVE_FIELD + MAX_SUPPORTED_BLOCK_SIZE;

std::atomic<bool> gStopFlag(false);
std::atomic<bool> bypassFlag(false);
void signalHandler(int) { gStopFlag = true; }

// A simple power-of-two ring buffer for single-producer / single-consumer
struct RingBuffer {
  std::vector<float> buf;
  size_t              mask;
  std::atomic<size_t> readIdx{0}, writeIdx{0};

  RingBuffer(size_t cap_pow2)
    : buf(cap_pow2), mask(cap_pow2-1) {}

  // how many samples are available to read
  size_t available() const {
    return (writeIdx.load(std::memory_order_acquire)
          - readIdx.load(std::memory_order_acquire)) & mask;
  }
  // how many samples you can still write
  size_t freeSpace() const {
    return (mask+1) - available();
  }

  // write up to n samples; returns how many were actually written
  size_t push(const float* data, size_t n) {
    size_t free = freeSpace();
    size_t toWrite = n < free ? n : free;
    size_t w0 = writeIdx.load(std::memory_order_relaxed);
    for(size_t i = 0; i < toWrite; ++i) {
      buf[(w0 + i) & mask] = data[i];
    }
    writeIdx.store((w0 + toWrite) & mask, std::memory_order_release);
    return toWrite;
  }

  // read up to n samples; returns how many were actually read
  size_t pop(float* out, size_t n) {
    size_t avail = available();
    size_t toRead = n < avail ? n : avail;
    size_t r0 = readIdx.load(std::memory_order_relaxed);
    for(size_t i = 0; i < toRead; ++i) {
      out[i] = buf[(r0 + i) & mask];
    }
    readIdx.store((r0 + toRead) & mask, std::memory_order_release);
    return toRead;
  }
};

// Bundles our two ring buffers
struct Shared {
  RingBuffer in, out;
  std::vector<float> tempOut;    // ← add this
  Shared(size_t cap_pow2)
    : in(cap_pow2), out(cap_pow2)
  {}
};

void inferenceThread(Shared* sh,
                     tflite::Interpreter* interpreter,
                     int hop) {
  // 1) build our sliding window
  std::vector<float> window(SEG_LEN);

  // 2) wait for the first full window
  while (!gStopFlag && sh->in.available() < (size_t)SEG_LEN) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (gStopFlag) return;

  // 3) read the first SEG_LEN samples
  sh->in.pop(window.data(), SEG_LEN);

  // temp space for the hop
  std::vector<float> newIn(hop);

  // 4) main loop
  while (!gStopFlag) {
    // 4a) pop exactly 'hop' samples (block until we get them)
    size_t got = 0;
    while (!gStopFlag && (got = sh->in.pop(newIn.data(), hop)) < (size_t)hop) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (gStopFlag) break;

    // 4b) slide the window
    std::memmove(window.data(),
                 window.data() + hop,
                 (SEG_LEN - hop) * sizeof(float));
    std::memcpy(window.data() + (SEG_LEN - hop),
                newIn.data(),
                hop * sizeof(float));

    // 4c) copy into TFLite
    float* inTensor = interpreter->typed_input_tensor<float>(0);
    std::memcpy(inTensor,
                window.data(),
                SEG_LEN * sizeof(float));

    // 4d) invoke
    if (interpreter->Invoke() != kTfLiteOk) {
      std::cerr << "TFLite invoke failed!\n";
    }

    // 4e) grab the last 'hop' samples of the output
    float* outTensor = interpreter->typed_output_tensor<float>(0);
    // tail-of-window is &outTensor[SEG_LEN - hop]
    sh->out.push(outTensor + (SEG_LEN - hop), hop);
    // if out buffer is full, excess data is simply dropped
  }
}

// This is the real-time PortAudio callback.
// It only pushes raw input into inRing and pops processed
// data from outRing (zero-fills on underrun), then stereo-duplicates.
static int paCallback(const void* inputBuffer,
                      void* outputBuffer,
                      unsigned long framesPerBuffer,
                      const PaStreamCallbackTimeInfo*,
                      PaStreamCallbackFlags,
                      void* userData)
{
  Shared* sh = static_cast<Shared*>(userData);
  const float* in  = static_cast<const float*>(inputBuffer);
  float*       out = static_cast<float*>(outputBuffer);
  size_t hop = framesPerBuffer;

  // 1) push raw input
  if (in) {
    sh->in.push(in, hop);
  }

  // bypass?
  if(bypassFlag.load()){
    if(in){
      for(size_t i=0;i<hop;i++){
        float v = in[i];
        out[2*i  ] = v;
        out[2*i+1] = v;
      }
    } else {
      memset(out,0,sizeof(float)*2*hop);
    }
    return paContinue;
  }

  // 2) pop processed output into our pre‐allocated tempOut
  float* temp = sh->tempOut.data();
  size_t got = sh->out.pop(temp, hop);
  if (got < hop) {
    // underrun → zero‐fill the rest
    memset(temp + got, 0, (hop - got) * sizeof(float));
  }

  // 3) stereo‐duplicate into the callback's output buffer
  for (size_t i = 0; i < hop; ++i) {
    out[2*i    ] = temp[i];
    out[2*i + 1] = temp[i];
  }

  return paContinue;
}

static void listDevicesAndExit() {
  Pa_Initialize();
  int nd = Pa_GetDeviceCount();
  if (nd < 0) {
    std::cerr << "PortAudio error: " << Pa_GetErrorText(nd) << "\n";
    Pa_Terminate();
    exit(1);
  }
  std::cout << "PortAudio devices:\n";
  for (int i = 0; i < nd; ++i) {
    const PaDeviceInfo* dpi = Pa_GetDeviceInfo(i);
    std::cout << "  [" << i << "] " << dpi->name
              << "   in="  << dpi->maxInputChannels
              << "   out=" << dpi->maxOutputChannels
              << "   sr="  << dpi->defaultSampleRate
              << "\n";
  }
  Pa_Terminate();
  exit(0);
}

int main(int argc, char* argv[]) {
  // -------- CLI parsing --------
  std::string modelPath;
  int    blockSize   = 4096;
  int    threads     = 4;
  int    inputDev    = -1;
  int    outputDev   = -1;
  bool   listDev     = false;

  for (int i = 1; i < argc; ++i) {
    if      (!strcmp(argv[i], "--model") && i+1<argc)     modelPath = argv[++i];
    else if (!strcmp(argv[i], "--blocksize") && i+1<argc) blockSize = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--threads") && i+1<argc)   threads   = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--input-device") && i+1<argc)  inputDev  = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--output-device")&& i+1<argc)  outputDev = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--list-devices"))          listDev   = true;
    else {
      std::cerr << "Unknown arg: " << argv[i] << "\n";
      return 1;
    }
  }

  if (listDev) {
    listDevicesAndExit();
  }
  if (modelPath.empty()) {
    std::cerr << "Error: --model <path> is required\n";
    return 1;
  }

  // -------- PortAudio setup --------
  if (Pa_Initialize() != paNoError) {
    std::cerr << "Pa_Initialize failed\n";
    return 1;
  }
  if (inputDev < 0 || outputDev < 0) {
    listDevicesAndExit();  // prints & exit
  }

  const PaDeviceInfo* diIn  = Pa_GetDeviceInfo(inputDev);
  const PaDeviceInfo* diOut = Pa_GetDeviceInfo(outputDev);
  if (!diIn || !diOut) {
    std::cerr << "Bad device index\n";
    Pa_Terminate();
    return 1;
  }
  int sampleRate = (int)std::lround(diIn->defaultSampleRate);

  PaStreamParameters inParams  = { inputDev,  1, paFloat32, diIn->defaultLowInputLatency,  nullptr };
  PaStreamParameters outParams = { outputDev, 2, paFloat32, diOut->defaultLowOutputLatency, nullptr };

  // -------- Load TFLite model and XNNPACK --------
  std::cout << "Loading model: " << modelPath << "\n";
  auto model = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
  if (!model) {
    std::cerr << "Failed to mmap model\n";
    Pa_Terminate();
    return 1;
  }
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (builder(&interpreter) != kTfLiteOk || !interpreter) {
    std::cerr << "Failed to construct interpreter\n";
    Pa_Terminate();
    return 1;
  }
  interpreter->SetNumThreads(threads);
  TfLiteXNNPackDelegateOptions xnnOpts = TfLiteXNNPackDelegateOptionsDefault();
  xnnOpts.num_threads = threads;
  TfLiteDelegate* xnnDelegate = TfLiteXNNPackDelegateCreate(&xnnOpts);
  if (interpreter->ModifyGraphWithDelegate(xnnDelegate) != kTfLiteOk) {
    std::cerr << "Warning: could not apply XNNPACK delegate\n";
  }

  // dynamic shape → fix it
  int inIdx = interpreter->inputs()[0];
  if (interpreter->ResizeInputTensor(inIdx, {1, SEG_LEN, 1}) != kTfLiteOk ||
      interpreter->AllocateTensors()                     != kTfLiteOk) {
    std::cerr << "Failed to resize/allocate tensors\n";
    Pa_Terminate();
    return 1;
  }

  std::cout << "Using " << threads << " threads; SEG_LEN=" << SEG_LEN
            << "  blockSize=" << blockSize << "  sr=" << sampleRate << "\n"
            << "InDevice  #" << inputDev  << " \"" << diIn->name  << "\"\n"
            << "OutDevice #" << outputDev << " \"" << diOut->name << "\"\n";

  // -------- Shared ring buffers --------
  size_t cap = 1;
  while(cap < (size_t)(RECEPTIVE_FIELD + blockSize)) cap <<= 1;

  Shared shared(cap);

  shared.tempOut.resize(blockSize);    // allocate once, here

    // switch stdin → nonblocking raw mode
  struct termios oldt, newt;
  tcgetattr(STDIN_FILENO,&oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON|ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);

  // -------- spawn inference thread --------
  std::signal(SIGINT, signalHandler);
  std::thread infThread(inferenceThread,
                        &shared,
                        interpreter.get(),
                        blockSize);

  // spawn keyboard thread for A/B toggle
  std::thread kbThread([&](){
    while(!gStopFlag){
      int c = getchar();
      if(c=='b'||c=='B'){
        bool now = !bypassFlag.load();
        bypassFlag.store(now);
        // flush stale model-output
        shared.out.readIdx.store(
          shared.out.writeIdx.load(std::memory_order_relaxed),
          std::memory_order_relaxed
        );
        std::cout<<"[MAIN] Bypass "<<(now?"ON":"OFF")<<"\n";
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });

  // -------- open & start audio stream --------
  PaStream* stream = nullptr;
  if (Pa_OpenStream(&stream,
                    &inParams,
                    &outParams,
                    sampleRate,
                    blockSize,
                    paNoFlag,
                    paCallback,
                    &shared) != paNoError)
  {
    std::cerr << "Pa_OpenStream failed\n";
    Pa_Terminate();
    return 1;
  }
  Pa_StartStream(stream);
  std::cout << "Streaming... press Ctrl+C to stop.\n";

  // -------- wait --------
  while (!gStopFlag) {
    Pa_Sleep(100);
  }

  // -------- shutdown --------
  std::cout << "\nStopping...\n";
  Pa_StopStream(stream);
  Pa_CloseStream(stream);
  Pa_Terminate();

  // join and cleanup
  infThread.join();
  kbThread.join();
  TfLiteXNNPackDelegateDelete(xnnDelegate);

  // restore stdin
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);

  std::cout << "Done.\n";
  return 0;
}