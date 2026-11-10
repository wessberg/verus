#pragma once
// Standard / system / PortAudio / TFLite includes shared by every module.

#include <iostream>
#include <vector>
#include <array>
#include <thread>
#include <atomic>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <csignal>
#include <cstring>
#include <cassert>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>
#include <cmath>
#include <sys/stat.h>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <iomanip>
#include <portaudio.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <limits>
#include <stdexcept>
#include <memory>
#include <cstdlib>
#include <cctype>
#include <cstdio>
#if defined(__linux__)
  #include <semaphore.h>
  #include <time.h>
#endif

// TensorFlow Lite
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#ifdef TFLITE_WITH_XNNPACK
  #include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#endif
