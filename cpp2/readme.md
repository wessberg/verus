# BUILD

**Preconditions**
You'll need bazelisk. With brew:

```
brew install bazelisk
```

You may need to uninstall and/or unlink bazel first

1. **Tensorflow**

Pick a stable version, e.g. 2.20.0

```
cd lib
git clone --depth 1 --branch v2.20.0 https://github.com/tensorflow/tensorflow.git tensorflow-src
```

Go to it and configure it:

```
cd tensorflow-src
python configure.py
```

If/when it asks for optimization flags to use during compilation, enter this on ARM mac:
`-Wno-sign-compare -O3 -mcpu=apple-m1`

- On Mac OS: Select No to CUDA, ROCm, Android and iOS
- On NVIDIA, Say yes to CUDA.
- On AMD, Say yes to ROCm

Now build with `bazelisk`:

```
# Build optimized, monolithic, with XNNPACK enabled
bazelisk build -c opt --config=monolithic --define tflite_with_xnnpack=true \
  //tensorflow/lite:libtensorflowlite.so
```

Make sure it is installed with XNNPACK enabled:

```
nm -gU /Users/ideanote/Developer/tf-audio-experiment/cpp2/lib/tensorflow-src/bazel-bin/tensorflow/lite/libtensorflowlite.so | grep TfLiteXNNPackDelegateCreate
```

If you see `_TfLiteXNNPackDelegateCreate` somewhere in there, then all is well!

2. **Flatbuffers**

Tensorflow requires v24.3.25 of Flatbuffers, so you must install that specific version:

```
cd lib
git clone --depth 1 --branch v24.3.25 https://github.com/google/flatbuffers.git flatbuffers
```

3. **Build the project**

Make sure you cd out into `/cpp2` before continuing.

```
mkdir -p build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTFLITE_INCLUDE_DIR=/Users/ideanote/Developer/tf-audio-experiment/cpp2/lib/tensorflow-src \
  -DTFLITE_LIBRARY=/Users/ideanote/Developer/tf-audio-experiment/cpp2/lib/tensorflow-src/bazel-bin/tensorflow/lite/libtensorflowlite.so \
  -DFLATBUFFERS_INCLUDE_DIR=/Users/ideanote/Developer/tf-audio-experiment/cpp2/lib/flatbuffers/include \
  -DCNPY_INCLUDE_DIR=/Users/ideanote/Developer/tf-audio-experiment/cpp2/lib \
  -DCNPY_LIBRARY=/Users/ideanote/Developer/tf-audio-experiment/cpp2/lib/cnpy/build/libcnpy.dylib \
  -DPORTAUDIO_INCLUDE_DIR=/Users/ideanote/Developer/tf-audio-experiment/cpp2/lib/portaudio/include \
  -DPORTAUDIO_LIBRARY=/usr/local/lib/libportaudio.dylib \
  -DUSE_TFLITE_XNNPACK=ON

cmake --build .
```

# USAGE

```
cmake --build . && ./stream_audio_cpp --tflite <model.tflite> --input-device <id> --output-device <id> [--blocksize N]
```

E.g.:

```
cmake --build . && ./stream_audio_cpp --tflite ../../out/models/realtime/piezo2mic_baseline_realtime_t32_fp32.tflite --input-device 4 --output-device 4 --blocksize 32
```