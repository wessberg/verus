/*
  Realtime TFLite-based mono-in / multi-out streamer with:
    - model-based piezo → mic processing,
    - time‑aligned DI,
    - stereo reverb / delay / chorus,
    - optional stereo FX loop,
    - headphone monitoring.

  Hardware / channel mapping
  ==========================

  ADC (input, 8 channels @ 48 kHz)
  --------------------------------
    ch1 (index 0) : Guitar IN (mono)
    ch2 (index 1) : Reserved for future stereo IN (right)
    ch3 (index 2) : FX RETURN L
    ch4 (index 3) : FX RETURN R
    ch5–ch8       : Unused (0.0)

  DAC (output, 8 channels @ 48 kHz)
  ---------------------------------
    ch1 (index 0) : MAIN OUT L
    ch2 (index 1) : MAIN OUT R
    ch3 (index 2) : DRY OUT (time‑aligned DI)
    ch4 (index 3) : FX SEND L  (pre‑FX-loop processed stereo)
    ch5 (index 4) : FX SEND R  (pre‑FX-loop processed stereo)
    ch6 (index 5) : HEADPHONE L (mirrors MAIN OUT L)
    ch7 (index 6) : HEADPHONE R (mirrors MAIN OUT R)
    ch8 (index 7) : MONO OUT (sum of MAIN L/R)

  High‑level signal flow
  ======================

  1. Input capture
     -------------
     - The PortAudio callback reads ADC ch1 (guitar IN) into a mono buffer `tempIn`.
     - `tempIn` is pushed into a ring buffer (`Shared::in`) for the inference thread.
     - At the same time, the callback deinterleaves:
         - ADC ch3 → `tempFxReturnL`  (FX RETURN L)
         - ADC ch4 → `tempFxReturnR`  (FX RETURN R)

  2. Inference thread (TFLite)
     --------------------------
     - The model is loaded, validated (hop == T for streaming models) and warmed up by main()
       BEFORE the stream starts; the inference thread only runs it.
     - Runs on a separate SCHED_FIFO thread (priority 35; the PortAudio callback thread sets
       itself to 40 on its first call — callback > inference > everything else).
     - Blocks on a semaphore posted by the PortAudio callback (no busy-spin), reads mono hops
       from `Shared::in`.
     - Feeds the model input `realtime_piezo_input` and the in-graph control knobs
       (main, pick attack, pick resonance, IR, dry/wet, output gain, phase, pickup type).
     - Produces:
         - Processed mono audio y[n] (piezo → mic),
         - Style probability (pick vs finger; already EMA-smoothed by the model),
         - Updated model states (streaming bundle).
     - Writes, AFTER the model ran, in lockstep (same hop, dry first):
         - time‑aligned dry mono x[n] → `Shared::dry` ring,
         - processed mono y[n] → `Shared::out` ring.
       The callback pops the same number of samples from both rings, so DI and processed
       audio can never drift apart, and trims any backlog beyond one hop so an xrun costs a
       one-hop dropout instead of permanently adding a hop of latency.
     - Never prints: telemetry goes into `gTelemetry` atomics and is printed by main().

  3. Audio callback post‑processing
     ------------------------------
     For each audio block (`framesPerBuffer = hop`):

     a) Pop from rings:
        - processed mono  → `tempOut` (proc)
        - time‑aligned DI → `tempDry` (dry)

     b) Insert processors on the processed mono path:
        - Master EQ (RBJ biquads: low cut 6/12/24/36 dB/oct, low shelf, mid bell, high shelf;
          identical to master_eq_sos() in src/verus/eq.py — the EQ is no longer part of the model)
        - Noise gate (side‑chain HPF, attack/hold/release, hysteresis, range)
        - Feedback protection frequency shifter (Hilbert-based)
        - Hall reverb:
            mono in  → stereo wet/dry mix → `tempStereo` (st[0]=L, st[1]=R)
          If reverb is disabled:
            `tempStereo` is just proc copied to L/R with appropriate dry/wet.
        - Stereo chorus (mono in → stereo wet add) on top of `tempStereo`
        - Tape/mod delay (mono in → stereo wet add) on top of `tempStereo`

        After this stage we have the internal **processed stereo**:
          preFxL = st[2*i + 0]
          preFxR = st[2*i + 1]

     c) FX loop routing
        ----------------
        - FX SEND (always active):
            DAC ch4 = FX SEND L = preFxL
            DAC ch5 = FX SEND R = preFxR

        - FX RETURN:
            ADC ch3 = FX RETURN L → tempFxReturnL
            ADC ch4 = FX RETURN R → tempFxReturnR

        - Software switch: gFxLoopEnabled (toggled from keyboard, can later
          be tied to jack detect / GPIO):

          If gFxLoopEnabled == false  (FX loop disabled):
            MAIN OUT and headphones use the internal processed stereo:

              mainL = preFxL
              mainR = preFxR

          If gFxLoopEnabled == true   (FX loop enabled):
            MAIN OUT and headphones use the external FX return:

              mainL = tempFxReturnL[i]
              mainR = tempFxReturnR[i]

        - DRY OUT:
            DAC ch3 (index 2) = dry[i]
          This is the time‑aligned DI coming from `Shared::dry`.

        - Headphones:
            DAC ch6 (index 5) = mainL
            DAC ch7 (index 6) = mainR
          Headphones always mirror the final MAIN OUT L/R after FX loop
          selection.

     d) Final DAC mapping per frame i:
        --------------------------------
          idx = 8 * i

          out[idx + 0] = mainL                  // MAIN OUT L (processed or FX return)
          out[idx + 1] = mainR                  // MAIN OUT R
          out[idx + 2] = dry[i]                 // DRY OUT (time‑aligned DI)
          out[idx + 3] = preFxL                 // FX SEND L (always internal processed)
          out[idx + 4] = preFxR                 // FX SEND R (always internal processed)
          out[idx + 5] = mainL                  // HEADPHONE L
          out[idx + 6] = mainR                  // HEADPHONE R
          out[idx + 7] = 0.5f * (mainL + mainR) // MONO OUT (MAIN L/R sum)

  4. Bypass behavior
     ----------------
     - Global bypass is controlled by gBypass (keyboard 'b').
     - On bypass:
         * All rings (in/out/dry) are flushed — by the PortAudio callback only (the single
           writer of the ring read pointers on the consumer side; other threads just bump
           gBypassEpoch).
         * Time‑domain FX (reverb, delay, chorus, gate, feedback shifter)
           are reset via their respective reset epochs.
         * The model keeps running on the live input while bypassed, so its streaming states
           are warm and un-bypassing is seamless (states are NOT reset on bypass edges).
         * The callback routes **dry mono input** from ADC ch1 to:

             MAIN L/R      : dry
             DRY OUT       : dry
             FX SEND L/R   : dry
             HEADPHONE L/R : dry

           (FX RETURN is ignored; gFxLoopEnabled is effectively irrelevant
            while bypass is active.)

  Notes
  =====
  - Sample rate is fixed to 48 kHz.
  - Rings are SPSC with monotonic counters; out/dry are only ever consumed (popped, trimmed or
    flushed) together by the same count, so DI and processed audio cannot drift apart.
  - The processed signal is sanitised (non-finite → 0, clamped to ±kProcSampleLimit) right after
    it is popped, before any stateful FX can latch a NaN.
  - Optional web UI: `--webui [port]` (127.0.0.1 only).
  - Streaming models REQUIRE hop == model T (--blocksize 32 for the exported T=32 models);
    windowed (non-streaming) models accept hop <= T.
  - All knob controls are atomic and can be changed at runtime via:
      * stdin keyboard shortcuts, and/or
      * the optional HTTP/JS Web UI (when enabled in main()).
*/

#include "common.hpp"
#include "audio/callback.hpp"
#include "audio/devices.hpp"
#include "console.hpp"
#include "fx/prewarm.hpp"
#include "model/inference_thread.hpp"
#include "ui/webui.hpp"

// ───────────────────────── MAIN ─────────────────────────
int main(int argc, char** argv) {
  int  blockSize = 32; // must equal the streaming model's T
  int  inDevId = -1, outDevId = -1;
  int  webUiPort = 0;  // 0 = web UI off
  int  latencyHops = 1; // processed hops queued ahead of the callback (jitter headroom vs latency)
  std::string tflitePath;

  bool listDev = false;
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"--list-devices")) listDev = true;
    else if(!strcmp(argv[i],"--blocksize") && i+1<argc) blockSize = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--input-device") && i+1<argc) inDevId  = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--output-device")&&i+1<argc) outDevId = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--tflite") && i+1<argc) tflitePath = argv[++i];
    else if(!strcmp(argv[i],"--latency-hops") && i+1<argc) latencyHops = std::max(1, std::min(8, atoi(argv[++i])));
    else if(!strcmp(argv[i],"--webui")) {
      webUiPort = 8088;
      if (i+1<argc && argv[i+1][0] != '-') webUiPort = atoi(argv[++i]);
    }
    else {
      std::cerr<<"Unknown arg "<<argv[i]<<"\n";
      return 1;
    }
  }

  auto devices = getDevices();
  if (listDev) { listDevices(devices); return 0; }

  if (inDevId<0 || outDevId<0 || tflitePath.empty() || blockSize <= 0) {
    std::cerr << "Usage: --tflite <model.tflite> --input-device <id> --output-device <id> [--blocksize N] "
                 "[--latency-hops N] [--webui [port]]\n";
    listDevices(devices);
    return 1;
  }

  bool haveIn = false, haveOut = false;
  for (auto &d : devices) {
    if (d.devId == inDevId)  haveIn  = (d.maxInputChannels  >= 8);
    if (d.devId == outDevId) haveOut = (d.maxOutputChannels >= 8);
  }
  if (!haveIn || !haveOut) {
    std::cerr<<"Invalid device IDs (need an existing input device with >= 8 inputs and an output device with >= 8 outputs)\n";
    listDevices(devices);
    return 1;
  }

  #ifdef __linux__
    // Lock current and future memory to avoid page faults in RT audio threads (the model below is
    // loaded AFTER this, so its weights/arena are locked too)
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall");
    }
    // NOTE: no process-wide SCHED_FIFO any more. The PortAudio callback and the inference thread
    // set their own priorities (kCallbackRtPriority > kInferenceRtPriority); main/keyboard/UI
    // threads stay SCHED_OTHER instead of inheriting an RT policy.
  #endif

  // Load, validate and warm up the model BEFORE any thread or stream exists: a bad model or a
  // wrong --blocksize fails cleanly here (instead of std::exit() inside the inference thread while
  // the stream was already running and the terminal in raw mode), and lazy first-invoke work does
  // not land in the first real-time hops.
  std::unique_ptr<TFLiteRealtime> model;
  try {
    model = std::make_unique<TFLiteRealtime>(tflitePath);
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  const int T = model->T();
  if (model->isStreaming() && blockSize != T) {
    // Streaming models REQUIRE hop == T: with hop < T the zero-padded tail of every block is fed
    // into the model states and the same samples are re-fed next hop — state corruption.
    std::cerr << "[MAIN] Error: this streaming model requires --blocksize " << T << " (got " << blockSize
              << "); a different hop corrupts the streaming states.\n";
    return 1;
  }
  if (!model->isStreaming() && blockSize > T) {
    std::cerr << "[MAIN] Error: blocksize (" << blockSize << ") must be <= T (" << T << ")\n";
    return 1;
  }
  if (!model->isStreaming() && blockSize != T)
    std::cerr << "[MAIN] WARN: hop (" << blockSize << ") != T (" << T << ") on a windowed model; sliding-window mode.\n";
  {
    auto t0 = std::chrono::steady_clock::now();
    model->warmUp();
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "[MAIN] model warm-up: " << std::chrono::duration<double, std::milli>(t1 - t0).count() / 8.0
              << " ms/hop (budget " << 1000.0 * blockSize / 48000.0 << " ms)\n";
  }

  PaError err = Pa_Initialize();
  if (err != paNoError) {
    std::cerr<<"Pa_Initialize failed: "<<Pa_GetErrorText(err)<<"\n";
    return 1;
  }
  auto *diIn  = Pa_GetDeviceInfo(inDevId);
  auto *diOut = Pa_GetDeviceInfo(outDevId);
  if (!diIn || !diOut) {
    std::cerr << "Pa_GetDeviceInfo failed for the selected devices\n";
    Pa_Terminate();
    return 1;
  }
  int sampleRate = 48000;

  std::cout<<"[MAIN] In ["<<inDevId<<"] "<<diIn->name<<" sr="<<diIn->defaultSampleRate<<"\n";
  std::cout<<"[MAIN] Out["<<outDevId<<"] "<<diOut->name<<" sr="<<diOut->defaultSampleRate<<"\n";
  std::cout<<"[MAIN] blockSize="<<blockSize<<"\n";
  std::cout<<"[MAIN] model: "<<tflitePath<<"\n";

  // PortAudio stream params
  double wantLatencySecs = double(blockSize) / sampleRate;
  PaStreamParameters ip = { inDevId, 8, paFloat32, wantLatencySecs, nullptr };
  PaStreamParameters op = { outDevId, 8, paFloat32, wantLatencySecs, nullptr };

  // stdin non-blocking for UI (only if stdin is a terminal; e.g. under systemd it is not)
  termios oldt{}, newt{};
  const bool haveTty = isatty(STDIN_FILENO) && tcgetattr(STDIN_FILENO,&oldt) == 0;
  if (haveTty) {
    newt = oldt; newt.c_lflag &= ~(ICANON|ECHO);
    tcsetattr(STDIN_FILENO,TCSANOW,&newt);
  }
  const int oldStdinFlags = fcntl(STDIN_FILENO, F_GETFL);
  fcntl(STDIN_FILENO,F_SETFL,O_NONBLOCK);
  auto restoreTerminal = [&](){
    if (haveTty) tcsetattr(STDIN_FILENO,TCSANOW,&oldt);
    if (oldStdinFlags >= 0) fcntl(STDIN_FILENO, F_SETFL, oldStdinFlags);
  };

  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);   // systemd / kill: shut down cleanly and restore the terminal

  // Shared rings (>= 16384 samples and >= 64 hops of headroom)
  size_t cap = nextPow2(std::max<size_t>(size_t(1) << 14, size_t(64) * size_t(blockSize)));
  Shared shared(cap);
  shared.hop = size_t(blockSize);
  shared.latencyHops = size_t(latencyHops);
  shared.tempIn.resize(blockSize);
  shared.tempOut.resize(blockSize);
  shared.tempStereo.resize(blockSize * 2);
  shared.tempDry.resize(blockSize);
  shared.tempFxReturnL.resize(blockSize);
  shared.tempFxReturnR.resize(blockSize);

  // Spawn inference thread
  InferCtx ictx;
  ictx.sh = &shared; ictx.hop = blockSize; ictx.sampleRate = sampleRate; ictx.rt = model.get();
  std::thread infThread([&](){ inferenceThread(ictx); });

  // Keyboard thread for bypass + knobs
  std::thread kbThread([&](){

    std::cout
    << "[MAIN] Keys: "
    << "b=Bypass  "
    << "q/a=PA +/-1dB  "
    << "w/s=PR +/-1dB  "
    << "e/d=IR +/-1dB  "
    << "r/f=Wet +/-5%  "
    << "t=Style mode (auto→pick→finger)  "
    << "g=Guided AUTO on/off  "
    << "l=FX loop on/off  "
    << "u=Pickup piezo/magnetic  "
    << "p=print"
    << "\n";

    while (!gStopFlag) {
      int c = getchar();
      if (c == EOF) { clearerr(stdin); std::this_thread::sleep_for(std::chrono::milliseconds(50)); continue; }
      if (c=='b'||c=='B') {
        bool now = !gBypass.load(std::memory_order_relaxed);
        gBypass.store(now, std::memory_order_release);
        gBypassEpoch.fetch_add(1, std::memory_order_acq_rel);   // the callback flushes the rings
        std::cout<<"[UI] Bypass "<<(now?"ON":"OFF")<<"\n";

      } else if (c=='l' || c=='L') {
        bool now = !gFxLoopEnabled.load(std::memory_order_relaxed);
        gFxLoopEnabled.store(now, std::memory_order_relaxed);
        std::cout << "[UI] FX loop "
                  << (now
                      ? "ENABLED (main outs/headphones from ADC ch3/4 return)"
                      : "DISABLED (main outs/headphones from internal processing)")
                  << "\n";

      } else if (c=='q') { gKnobs.pickAttack_dB.store( clampf(gKnobs.pickAttack_dB.load()+1.0f, -MAX_PICK_ATTACK_KNOB_CHANGE, MAX_PICK_ATTACK_KNOB_CHANGE) ); }
        else if (c=='a') { gKnobs.pickAttack_dB.store( clampf(gKnobs.pickAttack_dB.load()-1.0f, -MAX_PICK_ATTACK_KNOB_CHANGE, MAX_PICK_ATTACK_KNOB_CHANGE) ); }
        else if (c=='w') { gKnobs.pickReso_dB.store( clampf(gKnobs.pickReso_dB.load()+1.0f, -MAX_PICK_RESO_KNOB_CHANGE, MAX_PICK_RESO_KNOB_CHANGE) ); }
        else if (c=='s') { gKnobs.pickReso_dB.store( clampf(gKnobs.pickReso_dB.load()-1.0f, -MAX_PICK_RESO_KNOB_CHANGE, MAX_PICK_RESO_KNOB_CHANGE) ); }
        else if (c=='e') { gKnobs.ir_dB.store( clampf(gKnobs.ir_dB.load()+1.0f, -MAX_IR_KNOB_CHANGE, MAX_IR_KNOB_CHANGE) ); }
        else if (c=='d') { gKnobs.ir_dB.store( clampf(gKnobs.ir_dB.load()-1.0f, -MAX_IR_KNOB_CHANGE, MAX_IR_KNOB_CHANGE) ); }
        else if (c=='r') { gKnobs.wet_pct.store( clampf(gKnobs.wet_pct.load()+5.0f, 0.0f, 100.0f) ); }
        else if (c=='f') { gKnobs.wet_pct.store( clampf(gKnobs.wet_pct.load()-5.0f, 0.0f, 100.0f) ); }
        else if (c=='t') {
          int cur = gStyleMode.load();
          int next = (cur == static_cast<int>(StyleMode::Auto)) ? static_cast<int>(StyleMode::ForcePick)
                     : (cur == static_cast<int>(StyleMode::ForcePick)) ? static_cast<int>(StyleMode::ForceFinger)
                     : static_cast<int>(StyleMode::Auto);
          gStyleMode.store(next);
          std::cout<<"[UI] Style mode = "<<styleModeName(next)<<"\n";
        }
        else if (c=='p') {
          printKnobStatus("[UI]");
        }

        else if (c=='u' || c=='U') {
          float now = (gKnobs.pickup01.load() >= 0.5f) ? 0.0f : 1.0f;
          gKnobs.pickup01.store(now);
          std::cout << "[UI] Pickup = " << (now >= 0.5f ? "MAGNETIC" : "PIEZO") << "\n";
        }
        else if (c=='g' || c=='G') {
          bool now = !gGuidedAuto.load();
          gGuidedAuto.store(now);
          std::cout << "[UI] Guided AUTO style " << (now ? "ON (model style overridden by " + std::to_string((int)windowSec) + "s-stabilized prob)" : "OFF") << "\n";
        }
    }
  });

  if (webUiPort > 0) StartKnobWebUI(static_cast<uint16_t>(webUiPort));

  inputSignalInit();
  PrewarmDsp();   // allocate every FX/EQ processor before the first RT callback
  if (latencyHops > 1)
    std::cout << "[MAIN] latency-hops=" << latencyHops << " (+" << 1000.0 * (latencyHops - 1) * blockSize / sampleRate
              << " ms latency for " << latencyHops - 1 << " hop(s) of scheduling-jitter headroom)\n";

  auto shutdownThreads = [&](){
    gStopFlag = true;
    if (webUiPort > 0) StopKnobWebUI();
    infThread.join(); kbThread.join();
    restoreTerminal();
  };

  PaStream *stream = nullptr;
  err = Pa_OpenStream(&stream, &ip, &op, sampleRate, (unsigned long)blockSize, paNoFlag, paCallback, &shared);
  if (err != paNoError) {
    std::cerr<<"Pa_OpenStream failed: "<<Pa_GetErrorText(err)<<"\n";
    Pa_Terminate();
    shutdownThreads();
    return 1;
  }
  err = Pa_StartStream(stream);
  if (err != paNoError) {
    std::cerr<<"Pa_StartStream failed: "<<Pa_GetErrorText(err)<<"\n";
    Pa_CloseStream(stream); Pa_Terminate();
    shutdownThreads();
    return 1;
  }

  auto *info = Pa_GetStreamInfo(stream);
  if (info) {
    gIOInputLatencySec  = info->inputLatency;
    gIOOutputLatencySec = info->outputLatency;
  }
  std::cerr<<"[MAIN] Actual I/O buffer: in="<<(int)std::round(gIOInputLatencySec*sampleRate)
           <<" out="<<(int)std::round(gIOOutputLatencySec*sampleRate)<<" frames\n";
  
  StartTempMonitor(1000); // 1 Hz is plenty

  std::cout<<"[MAIN] Streaming… (Ctrl-C to quit)\n";
  // Telemetry is printed here (main thread) — never from the RT inference thread / callback.
  uint64_t seenPeriodic = 0, seenKnob = 0;
  bool warnedPrio = false, warnedXruns = false;
  while (!gStopFlag) {
    Pa_Sleep(50);
    const uint64_t ks = gTelemetry.knobSeq.load(std::memory_order_relaxed);
    if (ks != seenKnob) { seenKnob = ks; printKnobStatus("[UI]"); }
    const uint64_t ps = gTelemetry.periodicSeq.load(std::memory_order_relaxed);
    if (ps != seenPeriodic) { seenPeriodic = ps; printPeriodicTelemetry(); }
    if (!warnedPrio && gTelemetry.rtPrioFailed.load(std::memory_order_relaxed)) {
      warnedPrio = true;
      std::cerr << "[MAIN] WARNING: could not set SCHED_FIFO priorities (run with sudo / CAP_SYS_NICE); expect xruns.\n";
    }
    if (!warnedXruns && gTelemetry.outUnderruns.load(std::memory_order_relaxed) >= 20) {
      warnedXruns = true;
      std::cerr << "[MAIN] HINT: the inference thread keeps missing its " << 1000.0 * blockSize / sampleRate
                << " ms deadline (latency-hops=" << latencyHops << "). Each miss is a dropped hop (a click). "
                   "Try --latency-hops " << latencyHops + 1 << " (+" << 1000.0 * blockSize / sampleRate
                << " ms latency, one more period of headroom) or check the Proc= time above.\n";
    }
  }

  std::cout<<"[MAIN] Stopping…\n";

  Pa_StopStream(stream);
  Pa_CloseStream(stream);
  Pa_Terminate();
  StopTempMonitor();

  shutdownThreads();
  model.reset();   // after the inference thread is gone
  std::cout<<"[MAIN] Done.\n";
  return 0;
}
