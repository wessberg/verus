# BUILD

mkdir build
cd build
cmake .. \      
  -DTFLITE_INCLUDE_DIR=/Users/ideanote/Developer/tf-audio-experiment/cpp2/lib \
  -DTFLITE_LIBRARY=/Users/ideanote/Developer/tf-audio-experiment/cpp2/lib/tensorflow/bazel-bin/tensorflow/lite/libtensorflowlite.dylib \
  -DFLATBUFFERS_INCLUDE_DIR=/Users/ideanote/Developer/tf-audio-experiment/cpp2/lib
cmake --build .

# BUILD BENCHMARK

cd tensorflow-master
bazel-7.6.1 build -c opt tensorflow/lite/tools/benchmark:benchmark_model

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

cmake --build . && ./stream_audio_cpp --model ../model/piezo2mic_fp32.tflite --input-device 3 --output-device 3 --blocksize 8096 --buffer 0

# RUN BENCHMARK


./lib/tensorflow-master/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model \
  --graph=./model/piezo2mic_fp32.tflite \
  --num_threads=1 \
  --warmup_runs=5 \
  --num_runs=50 \
  --use_xnnpack=true