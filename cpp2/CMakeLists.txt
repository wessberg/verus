cmake_minimum_required(VERSION 3.16)
project(StreamAudioCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -march=native")

# Apple Silicon specifics
add_compile_definitions(
    ACCELERATE_NEW_LAPACK
    ACCELERATE_LAPACK_ILP64
)

# Dependencies
option(TFLITE_INCLUDE_DIR "Path to TensorFlow Lite include directory" "")
option(TFLITE_LIBRARY "Path to TensorFlow Lite library file (.dylib or .a)" "")
option(FLATBUFFERS_INCLUDE_DIR "Path to FlatBuffers include directory" "")
option(PORTAUDIO_INCLUDE_DIR "Path to Portaudio include directory" "")
option(PORTAUDIO_LIBRARY "Path to Portaudio library file" "")
option(CNPY_LIBRARY "Path to cnpy library file" "")
option(CNPY_INCLUDE_DIR "Path to cnpy include directory" "")

# Optional: XNNPACK delegate (TFLite C-API delegate library)
option(USE_TFLITE_XNNPACK "Enable XNNPACK delegate" OFF)
option(TFLITE_XNNPACK_LIBRARY "Path to TfLite XNNPACK delegate library (e.g., libtensorflowlite_delegate_xnnpack.dylib)" "")

if(NOT TFLITE_INCLUDE_DIR OR NOT TFLITE_LIBRARY OR NOT FLATBUFFERS_INCLUDE_DIR OR NOT CNPY_LIBRARY OR NOT CNPY_INCLUDE_DIR OR NOT PORTAUDIO_LIBRARY OR NOT PORTAUDIO_INCLUDE_DIR)
    message(FATAL_ERROR "You must specify TFLITE_INCLUDE_DIR, TFLITE_LIBRARY, FLATBUFFERS_INCLUDE_DIR, PORTAUDIO_LIBRARY, PORTAUDIO_INCLUDE_DIR, CNPY_LIBRARY and CNPY_INCLUDE_DIR.")
endif()

message(STATUS "Using TFLite include path: ${TFLITE_INCLUDE_DIR}")
message(STATUS "Using TFLite library: ${TFLITE_LIBRARY}")
if (USE_TFLITE_XNNPACK)
  message(STATUS "USE_TFLITE_XNNPACK=ON")
  if (TFLITE_XNNPACK_LIBRARY)
    message(STATUS "Using TFLite XNNPACK delegate library: ${TFLITE_XNNPACK_LIBRARY}")
  else()
    message(STATUS "No TFLITE_XNNPACK_LIBRARY provided; assuming libtensorflowlite already exports XNNPACK symbols.")
  endif()
else()
  message(STATUS "USE_TFLITE_XNNPACK=OFF (delegate code compiled out)")
endif()
message(STATUS "Using FlatBuffers include path: ${FLATBUFFERS_INCLUDE_DIR}")
message(STATUS "Using CNPY library: ${CNPY_LIBRARY}")
message(STATUS "Using CNPY include path: ${CNPY_INCLUDE_DIR}")
message(STATUS "Using Portaudio library: ${PORTAUDIO_LIBRARY}")
message(STATUS "Using Portaudio include path: ${PORTAUDIO_INCLUDE_DIR}")

add_executable(stream_audio_cpp src/main.cpp)

# Includes
target_include_directories(stream_audio_cpp PRIVATE
    "${TFLITE_INCLUDE_DIR}"
    "${FLATBUFFERS_INCLUDE_DIR}"
    "${PORTAUDIO_INCLUDE_DIR}"
    "${CNPY_INCLUDE_DIR}"
)

# Link libs
target_link_libraries(stream_audio_cpp PRIVATE
    "${TFLITE_LIBRARY}"
    "${PORTAUDIO_LIBRARY}"
    "${CNPY_LIBRARY}"
    "-framework Accelerate"
    "-framework CoreAudio"
    "-framework CoreFoundation"
)

# Optional XNNPACK delegate link + define
if (USE_TFLITE_XNNPACK)
  target_compile_definitions(stream_audio_cpp PRIVATE TFLITE_WITH_XNNPACK)
  if (TFLITE_XNNPACK_LIBRARY)
    target_link_libraries(stream_audio_cpp PRIVATE "${TFLITE_XNNPACK_LIBRARY}")
  endif()
endif()

# Threads (no 'dl' on macOS)
if (UNIX)
  find_package(Threads REQUIRED)
  target_link_libraries(stream_audio_cpp PRIVATE Threads::Threads)
  if (NOT APPLE)
    target_link_libraries(stream_audio_cpp PRIVATE dl)
  endif()
endif()

# RPATH so dylibs can be found when running from build dir
# Add RPATH entries for TFLite dylib dir and PortAudio dir if they are absolute paths
get_filename_component(TFLITE_LIB_DIR "${TFLITE_LIBRARY}" DIRECTORY)
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
  BUILD_RPATH "${rpaths}"
  INSTALL_RPATH "${rpaths}"
)