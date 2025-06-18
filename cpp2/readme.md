# BUILD

mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native" -DTFLITE_INCLUDE_DIR=/Users/ideanote/Developer/tf-audio-experiment/cpp2/lib -DTFLITE_LIBRARY=/Users/ideanote/Developer/tf-audio-experiment/cpp2/lib/tensorflow/bazel-bin/tensorflow/lite/libtensorflowlite.dylib -DFLATBUFFERS_INCLUDE_DIR=/Users/ideanote/Developer/tf-audio-experiment/cpp2/lib -DCNPY_INCLUDE_DIR=/Users/ideanote/Developer/tf-audio-experiment/cpp2/lib -DCNPY_LIBRARY=/Users/ideanote/Developer/tf-audio-experiment/cpp2/lib/cnpy/build/libcnpy.dylib
cmake --build .

# RUN

--weights-dir <path>: (Required) Path to the where weights are placed

--blocksize <int>: Processing block size (default: 4096).

--threads <int>: Number of threads for TFLite interpreter (default: 4).

--input-device <int>: Audio input device ID.

--output-device <int>: Audio output device ID.

--list-devices: List available audio devices and exit.

### Run

#### FP32

cmake --build . && ./stream_audio_cpp --weights-dir ../../out/weights --in-dev 2 --out-dev 2 --blocksize 64

#### INT8

cmake --build . && ./stream_audio_cpp --weights-dir ../../out/int8_weights --in-dev 3 --out-dev 3 --blocksize 64 --int8