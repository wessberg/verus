cmake_minimum_required(VERSION 3.16)
project(StreamAudioCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimize for Pi/ARM (tune as needed)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -march=native")

# Dependencies
option(TFLITE_INCLUDE_DIR "Path to TensorFlow Lite include directory" "")
option(TFLITE_LIBRARY      "Path to TensorFlow Lite library file (.so or .a)" "")
option(FLATBUFFERS_INCLUDE_DIR "Path to FlatBuffers include directory" "")
option(PORTAUDIO_INCLUDE_DIR   "Path to Portaudio include directory" "")
option(PORTAUDIO_LIBRARY       "Path to Portaudio library file" "")

option(USE_TFLITE_XNNPACK "Enable XNNPACK delegate" OFF)
option(TFLITE_XNNPACK_LIBRARY "Path to TfLite XNNPACK delegate library" "")

if(NOT TFLITE_INCLUDE_DIR OR NOT TFLITE_LIBRARY OR NOT FLATBUFFERS_INCLUDE_DIR
   OR NOT PORTAUDIO_LIBRARY OR NOT PORTAUDIO_INCLUDE_DIR)
  message(FATAL_ERROR "You must specify TFLITE_INCLUDE_DIR, TFLITE_LIBRARY, FLATBUFFERS_INCLUDE_DIR, PORTAUDIO_LIBRARY, PORTAUDIO_INCLUDE_DIR.")
endif()

message(STATUS "Using TFLite include path: ${TFLITE_INCLUDE_DIR}")
message(STATUS "Using TFLite library:      ${TFLITE_LIBRARY}")
message(STATUS "Using FlatBuffers include: ${FLATBUFFERS_INCLUDE_DIR}")
message(STATUS "Using Portaudio include:   ${PORTAUDIO_INCLUDE_DIR}")
message(STATUS "Using Portaudio library:   ${PORTAUDIO_LIBRARY}")

add_executable(stream_audio_cpp src/main.cpp)  # or stream_tflite.cpp

target_include_directories(stream_audio_cpp PRIVATE
  "${TFLITE_INCLUDE_DIR}"
  "${FLATBUFFERS_INCLUDE_DIR}"
  "${PORTAUDIO_INCLUDE_DIR}"
  "${CNPY_INCLUDE_DIR}"
)

target_link_libraries(stream_audio_cpp PRIVATE
  "${TFLITE_LIBRARY}"
  "${PORTAUDIO_LIBRARY}"
)

if (USE_TFLITE_XNNPACK)
  target_compile_definitions(stream_audio_cpp PRIVATE TFLITE_WITH_XNNPACK)
  if (TFLITE_XNNPACK_LIBRARY)
    target_link_libraries(stream_audio_cpp PRIVATE "${TFLITE_XNNPACK_LIBRARY}")
  endif()
endif()

# Threads
find_package(Threads REQUIRED)
target_link_libraries(stream_audio_cpp PRIVATE Threads::Threads)

# RPATH so .so libs can be found from build dir
get_filename_component(TFLITE_LIB_DIR    "${TFLITE_LIBRARY}"    DIRECTORY)
get_filename_component(PORTAUDIO_LIB_DIR "${PORTAUDIO_LIBRARY}" DIRECTORY)
set(rpaths "")
list(APPEND rpaths "${TFLITE_LIB_DIR}")
list(APPEND rpaths "${PORTAUDIO_LIB_DIR}")

if (TFLITE_XNNPACK_LIBRARY)
  get_filename_component(TFLITE_XNNPACK_LIB_DIR "${TFLITE_XNNPACK_LIBRARY}" DIRECTORY)
  list(APPEND rpaths "${TFLITE_XNNPACK_LIB_DIR}")
endif()

list(REMOVE_DUPLICATES rpaths)
set_target_properties(stream_audio_cpp PROPERTIES
  BUILD_RPATH   "${rpaths}"
  INSTALL_RPATH "${rpaths}"
)