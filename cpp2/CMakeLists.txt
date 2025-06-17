# CMake minimum version
cmake_minimum_required(VERSION 3.16)

# Project definition
project(StreamAudioCpp LANGUAGES CXX)

# Set C++ standard to 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependency Configuration ---

# 1. PortAudio (via pkg-config)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)

# 2. Manual Dependencies (TensorFlow Lite & FlatBuffers)
option(TFLITE_INCLUDE_DIR "Path to TensorFlow Lite include directory" "")
option(TFLITE_LIBRARY "Path to TensorFlow Lite library file" "")
option(FLATBUFFERS_INCLUDE_DIR "Path to FlatBuffers include directory" "")
option(CNPY_LIBRARY "Path to cnpy library file" "")
option(CNPY_INCLUDE_DIR "Path to cnpy include directory" "")

if(NOT TFLITE_INCLUDE_DIR OR NOT TFLITE_LIBRARY OR NOT FLATBUFFERS_INCLUDE_DIR OR NOT CNPY_LIBRARY OR NOT CNPY_INCLUDE_DIR)
    message(FATAL_ERROR "You must specify TFLITE_INCLUDE_DIR, TFLITE_LIBRARY, FLATBUFFERS_INCLUDE_DIR, CNPY_LIBRARY and CNPY_INCLUDE_DIR when configuring CMake.")
endif()

message(STATUS "Using TFLite include path: ${TFLITE_INCLUDE_DIR}")
message(STATUS "Using TFLite library: ${TFLITE_LIBRARY}")
message(STATUS "Using FlatBuffers include path: ${FLATBUFFERS_INCLUDE_DIR}")
message(STATUS "Using CNPY library: ${CNPY_LIBRARY}")
message(STATUS "Using CNPY include path: ${CNPY_INCLUDE_DIR}")

# Add executable
add_executable(stream_audio_cpp src/main.cpp)
add_executable(stream_audio_tflite_cpp src/main-tflite.cpp)

# --- Linking and Include Directories ---

# Include directories for the compiler
target_include_directories(stream_audio_cpp PRIVATE
    "${PORTAUDIO_INCLUDE_DIRS}"
    "${TFLITE_INCLUDE_DIR}"
    "${FLATBUFFERS_INCLUDE_DIR}"
    "${CNPY_INCLUDE_DIR}"
)

target_include_directories(stream_audio_tflite_cpp PRIVATE
    "${PORTAUDIO_INCLUDE_DIRS}"
    "${TFLITE_INCLUDE_DIR}"
    "${FLATBUFFERS_INCLUDE_DIR}"
)

# Explicitly tell the linker where to search for libraries
target_link_directories(stream_audio_cpp PRIVATE 
    "${PORTAUDIO_LIBRARY_DIRS}"
)

target_link_directories(stream_audio_tflite_cpp PRIVATE 
    "${PORTAUDIO_LIBRARY_DIRS}"
)

# Link the actual libraries
target_link_libraries(stream_audio_cpp PRIVATE
    # PORTAUDIO_LIBRARIES usually resolves to just the library name (e.g., "portaudio")
    # which target_link_directories helps the linker find.
    "${PORTAUDIO_LIBRARIES}"
    "${TFLITE_LIBRARY}"
    "${CNPY_LIBRARY}"
)

target_link_libraries(stream_audio_tflite_cpp PRIVATE
    # PORTAUDIO_LIBRARIES usually resolves to just the library name (e.g., "portaudio")
    # which target_link_directories helps the linker find.
    "${PORTAUDIO_LIBRARIES}"
    "${TFLITE_LIBRARY}"
)

# On Linux/macOS, you might need to link pthread and dl
if (UNIX)
    find_package(Threads REQUIRED)
    target_link_libraries(stream_audio_cpp PRIVATE Threads::Threads dl)
endif()

if (UNIX)
    find_package(Threads REQUIRED)
    target_link_libraries(stream_audio_tflite_cpp PRIVATE Threads::Threads dl)
endif()
