# BUILD

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native" -DTFLITE_INCLUDE_DIR=/Users/ideanote/Developer/tf-audio-experiment/cpp2/lib -DTFLITE_LIBRARY=/Users/ideanote/Developer/tf-audio-experiment/cpp2/lib/tensorflow/bazel-bin/tensorflow/lite/libtensorflowlite.dylib -DFLATBUFFERS_INCLUDE_DIR=/Users/ideanote/Developer/tf-audio-experiment/cpp2/lib -DCNPY_INCLUDE_DIR=/Users/ideanote/Developer/tf-audio-experiment/cpp2/lib -DCNPY_LIBRARY=/Users/ideanote/Developer/tf-audio-experiment/cpp2/lib/cnpy/build/libcnpy.dylib
cmake --build .

# RUN

cd build
./stream_audio_cpp --model ../model/piezo2mic_fp32.tflite

--model <path>: (Required) Path to the TFLite model file.

--blocksize <int>: Processing block size (default: 4096).

--buffer <float>: Buffer size in seconds (default: 1.0).

--threads <int>: Number of threads for TFLite interpreter (default: 4).

--input-device <int>: Audio input device ID.

--output-device <int>: Audio output device ID.

--list-devices: List available audio devices and exit.

### Run TFLIte streamer

cmake --build . && ./stream_audio_tflite_cpp --model ../../out/piezo2mic_int8.tflite --input-device 2 --output-device 2 --blocksize 256

### Run non-TFLite streamer

cmake --build . && ./stream_audio_cpp --weights-dir ../../out/weights --in-dev 2 --out-dev 2 --blocksize 256