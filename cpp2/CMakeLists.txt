# CMake minimum version
cmake_minimum_required(VERSION 3.16)

# Project definition
project(StreamAudioCpp LANGUAGES CXX)

# Set C++ standard to 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release flags: optimization + new Accelerate CBLAS headers
set(CMAKE_CXX_FLAGS_RELEASE   
    # Best for ARMv8-A, such Cortex-A
    #"-O3 -ffast-math -march=armv8-a"

    # Best for Apple Silicon
    "-O3 -ffast-math -march=native"
)

# These are also for Apple Silicon specifically
add_compile_definitions(
    ACCELERATE_NEW_LAPACK
    ACCELERATE_LAPACK_ILP64
)

# 2. Manual Dependencies (TensorFlow Lite & FlatBuffers)
option(TFLITE_INCLUDE_DIR "Path to TensorFlow Lite include directory" "")
option(TFLITE_LIBRARY "Path to TensorFlow Lite library file" "")
option(FLATBUFFERS_INCLUDE_DIR "Path to FlatBuffers include directory" "")
option(PORTAUDIO_INCLUDE_DIR "Path to Portaudio include directory" "")
option(PORTAUDIO_LIBRARY "Path to Portaudio library file" "")
option(CNPY_LIBRARY "Path to cnpy library file" "")
option(CNPY_INCLUDE_DIR "Path to cnpy include directory" "")

if(NOT TFLITE_INCLUDE_DIR OR NOT TFLITE_LIBRARY OR NOT FLATBUFFERS_INCLUDE_DIR OR NOT CNPY_LIBRARY OR NOT CNPY_INCLUDE_DIR  OR NOT PORTAUDIO_LIBRARY OR NOT PORTAUDIO_INCLUDE_DIR)
    message(FATAL_ERROR "You must specify TFLITE_INCLUDE_DIR, TFLITE_LIBRARY, FLATBUFFERS_INCLUDE_DIR, PORTAUDIO_LIBRARY, PORTAUDIO_INCLUDE_DIR, CNPY_LIBRARY and CNPY_INCLUDE_DIR when configuring CMake.")
endif()

message(STATUS "Using TFLite include path: ${TFLITE_INCLUDE_DIR}")
message(STATUS "Using TFLite library: ${TFLITE_LIBRARY}")
message(STATUS "Using FlatBuffers include path: ${FLATBUFFERS_INCLUDE_DIR}")
message(STATUS "Using CNPY library: ${CNPY_LIBRARY}")
message(STATUS "Using CNPY include path: ${CNPY_INCLUDE_DIR}")
message(STATUS "Using Portaudio library: ${PORTAUDIO_LIBRARY}")
message(STATUS "Using Portaudio include path: ${PORTAUDIO_INCLUDE_DIR}")

# Add executable
add_executable(stream_audio_cpp src/main.cpp)

# --- Linking and Include Directories ---

# Include directories for the compiler
target_include_directories(stream_audio_cpp PRIVATE
    "${PORTAUDIO_INCLUDE_DIR}"
    "${FLATBUFFERS_INCLUDE_DIR}"
    "${CNPY_INCLUDE_DIR}"
)

# Link the actual libraries
target_link_libraries(stream_audio_cpp PRIVATE
    # PORTAUDIO_LIBRARIES usually resolves to just the library name (e.g., "portaudio")
    # which target_link_directories helps the linker find.
    "${PORTAUDIO_LIBRARY}"
    "${CNPY_LIBRARY}"

    # This is for Apple Silicon specifically
     "-framework Accelerate"
     "-framework CoreAudio"
     "-framework CoreFoundation"
)

# On Linux/macOS, you might need to link pthread and dl
if (UNIX)
    find_package(Threads REQUIRED)
    target_link_libraries(stream_audio_cpp PRIVATE Threads::Threads dl)
endif()