# BUILD

**Preconditions**

You need to be running the PREEMPT RT Linux Kernel.
Run `uname -a` to verify that `PREEMPT_RT` is in there somewhere.

If it's not, follow this guide:
https://forum.linuxcnc.org/9-installing-linuxcnc/55048-raspberry-pi-os-preempt-rt-6-13-kernel-cookbook

Delete the linux folder afterwards - it takes up a lot of storage, enough that compiling tensorflow will eventually
run out of disk space.

Install these build dependencies:
```
sudo apt update
sudo apt install -y \
  git curl build-essential cmake pkg-config portaudio19-dev
  ```

1. **Bazelisk**
Install bazelisk:
```
cd $HOME
curl -L -o bazelisk https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-arm64
chmod +x bazelisk
sudo mv bazelisk /usr/local/bin/bazelisk

bazelisk --version
```

2. **Tensorflow**

Pick a stable version, e.g. 2.20.0
Building the full TensorFlow version will take up >4gb on disk,
but we only need TFLite for inferrence, so we'll make a minimal build.

```
cd $HOME
git clone --depth 1 --branch v2.20.0 https://github.com/tensorflow/tensorflow.git tensorflow
```

Go to it and configure it:

```
cd tensorflow
python configure.py
```

Say no to ROCm, CUDA, Clang, and Android.

When it asks for optimization flags, use:
```
-Wno-sign-compare -O3 -march=native
```

Now build with `bazelisk`:

```
bazelisk build -c opt --define tflite_with_xnnpack=true //tensorflow/lite:libtensorflowlite.so
sudo install -Dm755 bazel-bin/tensorflow/lite/libtensorflowlite.so /usr/local/lib/libtensorflowlite.so
sudo rsync -a tensorflow/ /usr/local/include/tensorflow/
sudo ldconfig
```

Make sure it is installed with XNNPACK enabled:

```
nm -D bazel-bin/tensorflow/lite/libtensorflowlite.so | grep TfLiteXNNPackDelegateCreate
```

If you see `TfLiteXNNPackDelegateCreate` somewhere in there, then all is well!

2. **Flatbuffers**

Tensorflow requires v24.3.25 of Flatbuffers, so you must install that specific version:

```
cd $HOME
git clone --depth 1 --branch v24.3.25 https://github.com/google/flatbuffers.git flatbuffers
```

3. **Build the project**

Make sure you cd out into `/cpp` before continuing.

```
mkdir -p build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTFLITE_INCLUDE_DIR=$HOME/tensorflow \
  -DTFLITE_LIBRARY=/usr/local/lib/libtensorflowlite.so \
  -DFLATBUFFERS_INCLUDE_DIR=$HOME/flatbuffers/include \
  -DPORTAUDIO_INCLUDE_DIR=/usr/include \
  -DPORTAUDIO_LIBRARY=/usr/lib/aarch64-linux-gnu/libportaudio.so \
  -DUSE_TFLITE_XNNPACK=ON

cmake --build .
```

# USAGE

```
cmake --build . && sudo ./stream_audio_cpp --tflite <model.tflite> --input-device <id> --output-device <id> [--blocksize N]
```

E.g.:

```
cmake --build . && sudo ./stream_audio_cpp --tflite ../../out/models/realtime/piezo2mic_baseline_realtime_t32_fp32.tflite --blocksize 32 --input-device 0 --output-device 0
```

Notes:

- `--blocksize` MUST equal the model's T (32 for the exported `_t32_` models); the binary refuses
  other values for streaming models because a smaller hop corrupts the streaming states.
- The master EQ (low cut / shelves / mid) runs inside this binary (`MasterEqIIR`, post-model),
  not in the model. The knobs are the existing EQ fields in `Knobs` / the (optional) web UI.
- Every two seconds the main thread prints processing time, style, and health counters:
  `xruns` (callback found no processed audio), `inDrops` (input ring full — the inference
  thread is too slow), `resyncs` (backlog trimmed after an xrun to keep latency at one hop).
  All three should stay at 0 during normal operation.
- The September 2026 changes (lockstep DI/processed rings, semaphore-driven inference thread,
  post-model EQ, DSP prewarm, main-thread telemetry) were syntax-checked against stub headers
  only; the first build on the Pi may need trivial fixes.
