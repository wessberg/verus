# Verus runtime

The realtime inference and playback engine: it captures the guitar from an audio interface, runs the
streaming TFLite model (32-sample hops), applies the post-model effects (master EQ, noise gate,
chorus, delay, reverb, feedback protection) and plays the result. Today it is a standalone
executable (`verus-runtime`) for the Raspberry Pi + HiFiBerry pedal. How it will be embedded in the
software that runs on the pedal, and in the desktop app and plugins, is not decided yet
([ROADMAP](../docs/ROADMAP.md)); this is why it is its own directory instead of part of the pedal
application.

## Source layout

`src/main.cpp` is the only translation unit; it includes the headers below. They define file-local
(`static`) state and are written to be included once, by `main.cpp`. If a header is ever shared
between translation units, make its file-scope `static`s `inline` first, or the state is duplicated.

| File | Holds |
|---|---|
| `main.cpp` | architecture notes (channel map, signal flow), command line, start-up and shutdown |
| `common.hpp` | the standard / system / PortAudio / TFLite includes |
| `state.hpp`, `knobs.hpp`, `telemetry.hpp` | process-wide flags and constants (incl. the pick-branch baselines that mirror `src/verus/config.py`); the atomic runtime controls; RT-thread counters |
| `realtime.hpp`, `util.hpp`, `soc_temp.hpp`, `prob_window.hpp` | SCHED_FIFO / flush-to-zero / input signal; numeric helpers; Pi temperature; style-probability window |
| `model/tflite_realtime.hpp`, `model/inference_thread.hpp` | the TFLite wrapper (state bundle, I/O resolution); the inference thread |
| `audio/callback.hpp`, `audio/ring_buffer.hpp`, `audio/devices.hpp` | the PortAudio callback; SPSC rings and shared state; device enumeration |
| `fx/*.hpp` | master EQ (`master_eq.hpp`, mirrors `src/verus/eq.py`), noise gate, delay, chorus, reverb, feedback protection, prewarm |
| `ui/webui.hpp`, `ui/webui_html.hpp` | the `--webui` HTTP server and its page |
| `console.hpp` | status and telemetry printing |

# Build
**Preconditions**

You need to be running the PREEMPT RT Linux Kernel.
Run `uname -a` to verify that `PREEMPT_RT` is in there somewhere.

If it's not, follow this guide:
https://forum.linuxcnc.org/9-installing-linuxcnc/55048-raspberry-pi-os-preempt-rt-6-13-kernel-cookbook

Delete the linux folder afterwards - it takes up a lot of storage, enough that compiling tensorflow will eventually
run out of disk space.

Install these build dependencies:
```
sudo apt update
sudo apt install -y \
  git curl build-essential cmake pkg-config portaudio19-dev
  ```

1. **Bazelisk**
Install bazelisk:
```
cd $HOME
curl -L -o bazelisk https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-arm64
chmod +x bazelisk
sudo mv bazelisk /usr/local/bin/bazelisk

bazelisk --version
```

2. **Tensorflow**

Pick a stable version, e.g. 2.20.0
Building the full TensorFlow version will take up >4gb on disk,
but we only need TFLite for inferrence, so we'll make a minimal build.

```
cd $HOME
git clone --depth 1 --branch v2.20.0 https://github.com/tensorflow/tensorflow.git tensorflow
```

Go to it and configure it:

```
cd tensorflow
python configure.py
```

Say no to ROCm, CUDA, Clang, and Android.

When it asks for optimization flags, use:
```
-Wno-sign-compare -O3 -march=native
```

Now build with `bazelisk`:

```
bazelisk build -c opt --define tflite_with_xnnpack=true //tensorflow/lite:libtensorflowlite.so
sudo install -Dm755 bazel-bin/tensorflow/lite/libtensorflowlite.so /usr/local/lib/libtensorflowlite.so
sudo rsync -a tensorflow/ /usr/local/include/tensorflow/
sudo ldconfig
```

Make sure it is installed with XNNPACK enabled:

```
nm -D bazel-bin/tensorflow/lite/libtensorflowlite.so | grep TfLiteXNNPackDelegateCreate
```

If you see `TfLiteXNNPackDelegateCreate` somewhere in there, then all is well!

2. **Flatbuffers**

Tensorflow requires v24.3.25 of Flatbuffers, so you must install that specific version:

```
cd $HOME
git clone --depth 1 --branch v24.3.25 https://github.com/google/flatbuffers.git flatbuffers
```

3. **Build the project**

Make sure you cd out into `/runtime` before continuing.

```
mkdir -p build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTFLITE_INCLUDE_DIR=$HOME/tensorflow \
  -DTFLITE_LIBRARY=/usr/local/lib/libtensorflowlite.so \
  -DFLATBUFFERS_INCLUDE_DIR=$HOME/flatbuffers/include \
  -DPORTAUDIO_INCLUDE_DIR=/usr/include \
  -DPORTAUDIO_LIBRARY=/usr/lib/aarch64-linux-gnu/libportaudio.so \
  -DUSE_TFLITE_XNNPACK=ON

cmake --build .
```

# USAGE

```
cmake --build . && sudo ./verus-runtime --tflite <model.tflite> --input-device <id> --output-device <id> [--blocksize N] [--latency-hops N] [--webui [port]]
```

E.g. with the model written by `verus --export-tflite`:

```
cmake --build . && sudo ./verus-runtime --tflite ../../out/models/realtime/verus_baseline_realtime_t32_fp32.tflite --blocksize 32 --input-device 0 --output-device 0
```

Notes:

- `--blocksize` MUST equal the model's T (32 for the exported `_t32_` models); the binary refuses
  other values for streaming models because a smaller hop corrupts the streaming states.
- The master EQ (low cut / shelves / mid) runs inside this binary (`MasterEqIIR`, post-model),
  not in the model. The knobs are the existing EQ fields in `Knobs` / the (optional) web UI.
- Every two seconds the main thread prints processing time, style, and health counters:
  `xruns` (callback found no processed audio → a dropped hop), `paXruns` (PortAudio/ALSA
  reported an overflow/underflow), `inDrops` (input ring full — the inference thread is too
  slow), `resyncs` (backlog trimmed after a stall to keep the latency at `--latency-hops`).
  All of them should stay at 0 during normal operation.
- At start-up the model is loaded, validated and warmed up before the stream opens; the warm-up
  prints the measured ms/hop next to the 0.67 ms budget. If `Proc=` is close to the budget or
  `xruns` climb, use `--latency-hops 2` (+0.67 ms latency, one extra period of headroom).
- Run with `sudo` (or CAP_SYS_NICE): the callback thread uses SCHED_FIFO 40 and the inference
  thread SCHED_FIFO 35; a warning is printed if that is not permitted.
- `--webui [port]` serves the knob UI on 127.0.0.1 (default port 8088).
- Build verification: before the source was split into headers (2026-09-22) it compiled
  warning-free (`-Wall -Wextra`) with GCC 12.2 on Debian Bookworm arm64 (the Raspberry Pi OS
  toolchain) against the real TFLite 2.20, flatbuffers 24.3.25 and PortAudio headers, and its
  TFLite wrapper, master EQ, ring buffers and callback/inference pipeline were exercised in a
  native test harness (outputs bit-identical to the Python TFLite runner). The split was checked
  against stub TFLite / PortAudio headers only: every header compiles on its own and `main.cpp`
  produces the same machine code as the single file. Repeat the Docker build with the real
  headers before trusting it. It has not been run on real audio hardware yet.
- The CMake file defaults to a Release build. It keeps `-ffast-math` but adds
  `-fno-finite-math-only`; without it the compiler deletes every `std::isfinite` NaN guard.
