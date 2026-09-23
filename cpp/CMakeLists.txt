cmake_minimum_required(VERSION 3.16)
project(StreamAudioCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to an optimized build (the RT loop has ~0.67 ms per hop)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Optimize for Pi/ARM (tune as needed). -ffast-math implies -ffinite-math-only, which lets the
# compiler fold every std::isfinite() check to 'true' — the NaN/Inf guards in the FX/EQ and on the
# model output would silently disappear. -fno-finite-math-only keeps them (cost: nothing measurable).
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -fno-finite-math-only -march=native")

# Dependencies (paths: CACHE PATH/FILEPATH, not option(), which is for ON/OFF switches)
set(TFLITE_INCLUDE_DIR      "" CACHE PATH     "Path to TensorFlow Lite include directory (TF source root)")
set(TFLITE_LIBRARY          "" CACHE FILEPATH "Path to TensorFlow Lite library file (.so or .a)")
set(FLATBUFFERS_INCLUDE_DIR "" CACHE PATH     "Path to FlatBuffers include directory")
set(PORTAUDIO_INCLUDE_DIR   "" CACHE PATH     "Path to Portaudio include directory")
set(PORTAUDIO_LIBRARY       "" CACHE FILEPATH "Path to Portaudio library file")

option(USE_TFLITE_XNNPACK "Enable XNNPACK delegate" OFF)
set(TFLITE_XNNPACK_LIBRARY "" CACHE FILEPATH "Path to TfLite XNNPACK delegate library (only if not inside libtensorflowlite)")

if(NOT TFLITE_INCLUDE_DIR OR NOT TFLITE_LIBRARY OR NOT FLATBUFFERS_INCLUDE_DIR
   OR NOT PORTAUDIO_LIBRARY OR NOT PORTAUDIO_INCLUDE_DIR)
  message(FATAL_ERROR "You must specify TFLITE_INCLUDE_DIR, TFLITE_LIBRARY, FLATBUFFERS_INCLUDE_DIR, PORTAUDIO_LIBRARY, PORTAUDIO_INCLUDE_DIR.")
endif()

message(STATUS "Using TFLite include path: ${TFLITE_INCLUDE_DIR}")
message(STATUS "Using TFLite library:      ${TFLITE_LIBRARY}")
message(STATUS "Using FlatBuffers include: ${FLATBUFFERS_INCLUDE_DIR}")
message(STATUS "Using Portaudio include:   ${PORTAUDIO_INCLUDE_DIR}")
message(STATUS "Using Portaudio library:   ${PORTAUDIO_LIBRARY}")

add_executable(stream_audio_cpp src/main.cpp)

target_include_directories(stream_audio_cpp PRIVATE
  "${TFLITE_INCLUDE_DIR}"
  "${FLATBUFFERS_INCLUDE_DIR}"
  "${PORTAUDIO_INCLUDE_DIR}"
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