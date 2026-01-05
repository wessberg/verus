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
# Build optimized, monolithic, with XNNPACK enabled
bazelisk build -c opt --config=monolithic --define tflite_with_xnnpack=true \
  //tensorflow/lite:libtensorflowlite.so
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
  -DTFLITE_LIBRARY=$HOME/tensorflow/bazel-bin/tensorflow/lite/libtensorflowlite.so \
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