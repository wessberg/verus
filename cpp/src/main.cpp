// stream_piezo2mic.cpp
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <atomic>
#include <csignal>
#include <cmath>
#include <portaudio.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

// The model was trained on SEG_LEN=8192*2=16384 samples
static constexpr int SEG_LEN = 8192 * 2;

std::atomic<bool> gStopFlag(false);
void signalHandler(int s) {
    (void)s;
    gStopFlag = true;
}

struct UserData {
    tflite::Interpreter* interpreter = nullptr;
    int seg_len = 0;
    int hop_size = 0;
    int capacity = 0;
    std::vector<float> inputBuffer;
    std::vector<float> overlapBuffer;
    std::vector<float> weightBuffer;
    int writePtr = 0;
    int readPtr = 0;
};

static int paCallback(const void* inputBuffer,
                      void* outputBuffer,
                      unsigned long framesPerBuffer,
                      const PaStreamCallbackTimeInfo* /*timeInfo*/,
                      PaStreamCallbackFlags /*statusFlags*/,
                      void* userData)
{
    UserData* d = static_cast<UserData*>(userData);
    const float* in = static_cast<const float*>(inputBuffer);
    float* out = static_cast<float*>(outputBuffer);
    int hop   = d->hop_size;
    int seg   = d->seg_len;
    int cap   = d->capacity;

    // 1) shift old input, append new (or zeros if underflow)
    memmove(d->inputBuffer.data(),
            d->inputBuffer.data() + hop,
            (seg - hop) * sizeof(float));
    if (inputBuffer) {
        memcpy(d->inputBuffer.data() + (seg - hop),
               in,
               hop * sizeof(float));
    } else {
        // underflow: fill with silence
        memset(d->inputBuffer.data() + (seg - hop),
               0,
               hop * sizeof(float));
    }

    // 2) copy into TFLite input tensor
    float* inTensor = d->interpreter->typed_input_tensor<float>(0);
    memcpy(inTensor, d->inputBuffer.data(), seg * sizeof(float));

    // 3) invoke
    if (d->interpreter->Invoke() != kTfLiteOk) {
        // In a real app you might flag an error
    }

    // 4) grab output segment of length seg
    float* outTensor = d->interpreter->typed_output_tensor<float>(0);

    // 5) overlap-add into our ring
    int wptr = d->writePtr;
    for (int i = 0; i < seg; ++i) {
        int idx = wptr + i;
        if (idx >= cap) idx -= cap;  // single wrap is enough since cap >= seg
        d->overlapBuffer[idx] += outTensor[i];
        d->weightBuffer[idx]  += 1.0f;
    }

    // 6) read hop samples, normalize & stereo-duplicate
    int rptr = d->readPtr;
    for (int i = 0; i < hop; ++i) {
        int idx = rptr + i;
        if (idx >= cap) idx -= cap;
        float sum = d->overlapBuffer[idx];
        float w   = d->weightBuffer[idx];
        float v   = (w > 0.0f ? sum / w : 0.0f);
        // clear for next time
        d->overlapBuffer[idx] = 0.0f;
        d->weightBuffer[idx]  = 0.0f;
        // stereo out
        out[2*i    ] = v;
        out[2*i + 1] = v;
    }

    // advance pointers
    d->writePtr = wptr + hop; if (d->writePtr >= cap) d->writePtr -= cap;
    d->readPtr  = rptr + hop; if (d->readPtr  >= cap) d->readPtr  -= cap;

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
                  << "   maxIn="  << dpi->maxInputChannels
                  << "   maxOut=" << dpi->maxOutputChannels
                  << "   sr="     << dpi->defaultSampleRate
                  << "\n";
    }
    Pa_Terminate();
    exit(0);
}

int main(int argc, char* argv[]) {
    // --- parse CLI ---
    std::string modelPath;
    int    blockSize    = 4096;
    float  bufferSec    = 1.0f;
    int    threads      = 4;
    int    inputDev     = -1;
    int    outputDev    = -1;
    bool   wantListDev  = false;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--model") && i+1<argc) {
            modelPath = argv[++i];
        }
        else if (!strcmp(argv[i], "--blocksize") && i+1<argc) {
            blockSize = std::atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "--buffer") && i+1<argc) {
            bufferSec = std::atof(argv[++i]);
        }
        else if (!strcmp(argv[i], "--threads") && i+1<argc) {
            threads = std::atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "--input-device") && i+1<argc) {
            inputDev = std::atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "--output-device") && i+1<argc) {
            outputDev = std::atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "--list-devices")) {
            wantListDev = true;
        }
        else {
            std::cerr << "Unknown arg: " << argv[i] << "\n";
            return 1;
        }
    }

    if (wantListDev) {
        listDevicesAndExit();
    }
    if (modelPath.empty()) {
        std::cerr << "Error: --model <path> is required\n";
        return 1;
    }

    // --- init PortAudio ---
    PaError paErr = Pa_Initialize();
    if (paErr != paNoError) {
        std::cerr << "Pa_Initialize error: " << Pa_GetErrorText(paErr) << "\n";
        return 1;
    }

    // choose devices: if not given on CLI, list & ask interactively
    int defaultIn  = Pa_GetDefaultInputDevice();
    int defaultOut = Pa_GetDefaultOutputDevice();
    if (inputDev < 0 || outputDev < 0) {
        // list all PortAudio devices
        std::cout << "Available PortAudio devices:\n";
        int nd = Pa_GetDeviceCount();
        for (int i = 0; i < nd; ++i) {
            const PaDeviceInfo* dpi = Pa_GetDeviceInfo(i);
            std::cout << "  [" << i << "] " << dpi->name
                      << "   maxIn="  << dpi->maxInputChannels
                      << "   maxOut=" << dpi->maxOutputChannels
                      << "   sr="     << dpi->defaultSampleRate
                      << "\n";
        }
        // prompt the user
        std::cout << "Enter input device ID: ";
        std::cin  >> inputDev;
        std::cout << "Enter output device ID: ";
        std::cin  >> outputDev;
    }


    const PaDeviceInfo* diIn  = Pa_GetDeviceInfo(inputDev);
    const PaDeviceInfo* diOut = Pa_GetDeviceInfo(outputDev);
    if (!diIn || !diOut) {
        std::cerr << "Error: bad device index\n";
        Pa_Terminate();
        return 1;
    }

    // we'll use the input device's default sample rate
    int sampleRate = static_cast<int>(std::lround(diIn->defaultSampleRate));

    PaStreamParameters inParams = {};
    inParams.device            = inputDev;
    inParams.channelCount      = 1;
    inParams.sampleFormat      = paFloat32;
    inParams.suggestedLatency  = diIn->defaultLowInputLatency;
    inParams.hostApiSpecificStreamInfo = nullptr;

    PaStreamParameters outParams = {};
    outParams.device            = outputDev;
    outParams.channelCount      = 2;
    outParams.sampleFormat      = paFloat32;
    outParams.suggestedLatency  = diOut->defaultLowOutputLatency;
    outParams.hostApiSpecificStreamInfo = nullptr;

    // --- load TFLite model ---
    std::cout << "Loading TFLite model from: " << modelPath << "\n";
    auto model = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
    if (!model) {
        std::cerr << "Failed to mmap model\n";
        Pa_Terminate();
        return 1;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to construct interpreter\n";
        Pa_Terminate();
        return 1;
    }
    interpreter->SetNumThreads(threads);

    // XNNPACK delegate
    TfLiteXNNPackDelegateOptions xnnOpts = TfLiteXNNPackDelegateOptionsDefault();
    xnnOpts.num_threads = threads;
    TfLiteDelegate* xnnDelegate = TfLiteXNNPackDelegateCreate(&xnnOpts);
    if (interpreter->ModifyGraphWithDelegate(xnnDelegate) != kTfLiteOk) {
        std::cerr << "Failed to apply XNNPACK delegate\n";
        // we'll continue but performance may suffer
    }

    // dynamic-shape model → fix to SEG_LEN
    int inIdx = interpreter->inputs()[0];
    if (interpreter->ResizeInputTensor(inIdx, {1, SEG_LEN, 1}) != kTfLiteOk ||
        interpreter->AllocateTensors()                     != kTfLiteOk) {
        std::cerr << "Failed to resize+allocate tensors\n";
        Pa_Terminate();
        return 1;
    }

    // log some model info
    const TfLiteTensor* itensor = interpreter->tensor(inIdx);
    std::cout << "Interpreter inputs=" << interpreter->inputs().size()
              << "   input dims:";
    for (int i = 0; i < itensor->dims->size; ++i) {
        std::cout << " " << itensor->dims->data[i];
    }
    std::cout << "\nUsing " << threads << " TFLite threads with XNNPACK\n";

    std::cout << "Input device  [" << inputDev  << "] " << diIn->name << "\n"
              << "Output device [" << outputDev << "] " << diOut->name << "\n"
              << "Sample rate: " << sampleRate << "  BlockSize: " << blockSize << "\n";

    // --- prepare overlap-add buffers ---
    UserData data;
    data.interpreter = interpreter.get();
    data.seg_len     = SEG_LEN;
    data.hop_size    = blockSize;
    int bufferFrames = static_cast<int>(bufferSec * sampleRate + 0.5f);
    // round bufferFrames up to multiple of hop
    bufferFrames = ((bufferFrames + blockSize - 1)/blockSize)*blockSize;
    data.capacity    = SEG_LEN + bufferFrames;
    if (data.capacity < SEG_LEN + blockSize)
        data.capacity = SEG_LEN + blockSize;
    data.inputBuffer   .assign(SEG_LEN, 0.0f);
    data.overlapBuffer .assign(data.capacity, 0.0f);
    data.weightBuffer  .assign(data.capacity, 0.0f);
    data.writePtr = data.readPtr = 0;

    // --- open stream ---
    PaStream* stream = nullptr;
    paErr = Pa_OpenStream(&stream,
                          &inParams,
                          &outParams,
                          sampleRate,
                          blockSize,
                          paNoFlag,
                          paCallback,
                          &data);
    if (paErr != paNoError) {
        std::cerr << "Pa_OpenStream error: " << Pa_GetErrorText(paErr) << "\n";
        Pa_Terminate();
        return 1;
    }

    // catch Ctrl+C
    std::signal(SIGINT, signalHandler);

    paErr = Pa_StartStream(stream);
    if (paErr != paNoError) {
        std::cerr << "Pa_StartStream error: " << Pa_GetErrorText(paErr) << "\n";
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }
    std::cout << "Streaming... Press Ctrl+C to stop.\n";

    // block until user stops
    while (!gStopFlag) {
        Pa_Sleep(100);
    }

    std::cout << "\nStopping...\n";
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    // clean up delegate
    if (xnnDelegate) {
        TfLiteXNNPackDelegateDelete(xnnDelegate);
    }

    std::cout << "Done.\n";
    return 0;
}