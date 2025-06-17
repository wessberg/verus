// stream_piezo2mic_ab.cpp

#include <iostream>
#include <vector>
#include <array>
#include <thread>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <cassert>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>
#include <portaudio.h>
#include "cnpy.h"

// ───────────────────────── PARAMETERS ─────────────────────────
static constexpr int FILTERS   = 32;
static constexpr int KERNEL    = 63;
static constexpr std::array<int,6> DILATIONS = {1,2,4,8,16,32};
static constexpr int NUM_BLOCKS = (int)DILATIONS.size();
static constexpr int MAX_HOP    = 512;
static constexpr int RECEPTIVE_FIELD =
  1 + (KERNEL - 1) * (1+2+4+8+16+32);

// ───────────────────────── GLOBAL FLAGS ───────────────────────
std::atomic<bool> gStopFlag(false);
std::atomic<bool> bypassFlag(false);
void signalHandler(int){ gStopFlag = true; }

// ───────────────────────── RING BUFFER ────────────────────────
struct RingBuffer {
  std::vector<float> buf;
  size_t             mask;
  std::atomic<size_t> readIdx{0}, writeIdx{0};

  RingBuffer(size_t cap_pow2)
    : buf(cap_pow2), mask(cap_pow2-1) {}

  size_t available() const {
    return (writeIdx.load(std::memory_order_acquire)
          - readIdx .load(std::memory_order_acquire)) & mask;
  }
  size_t freeSpace() const {
    return (mask+1) - available();
  }

  // push up to n samples, return how many written
  size_t push(const float* data, size_t n) {
    size_t free = freeSpace(), toWrite = std::min(n, free);
    size_t w0   = writeIdx.load(std::memory_order_relaxed);
    for(size_t i=0;i<toWrite;i++)
      buf[(w0 + i)&mask] = data[i];
    writeIdx.store((w0 + toWrite)&mask,
                   std::memory_order_release);
    return toWrite;
  }

  // pop up to n samples, return how many read
  size_t pop(float* out, size_t n) {
    size_t av    = available(),
           toRead= std::min(n, av);
    size_t r0    = readIdx.load(std::memory_order_relaxed);
    for(size_t i=0;i<toRead;i++)
      out[i] = buf[(r0 + i)&mask];
    readIdx.store((r0 + toRead)&mask,
                  std::memory_order_release);
    return toRead;
  }
};

// Bundles in/out rings + temp vector for callback
struct Shared {
  RingBuffer in, out;
  std::vector<float> tempOut;
  Shared(size_t cap): in(cap), out(cap){}
};

// ───────────────────────── STREAMING CONV KERNELS ─────────────────────
// 1×1 pointwise conv
struct P1D {
  int inC, outC;
  std::vector<float> W, b;  // W: outC*inC,  b:outC

  P1D(){}
  P1D(int _inC,int _outC,
      std::vector<float>&& _W,
      std::vector<float>&& _b)
    : inC(_inC), outC(_outC), W(std::move(_W)), b(std::move(_b))
  {
    assert((int)W.size() == inC*outC);
    assert((int)b.size() == outC);
  }

  // in:  hop×inC  row-major
  // out: hop×outC row-major
  void run(const float* in, float* out, int hop) const {
    for(int t=0; t<hop; t++){
      const float* inp = in  + t*inC;
      float*       outp= out + t*outC;
      for(int oc=0; oc<outC; oc++){
        float acc = b[oc];
        const float* wrow = &W[oc*inC];
        for(int ic=0; ic<inC; ic++)
          acc += wrow[ic]*inp[ic];
        outp[oc] = acc;
      }
    }
  }
};

// Depthwise dilated conv with circular state buffer
struct D1D {
  int C,K,D,bufL;
  std::vector<float> W,b, buf;
  int head=0;

  D1D(){}
  D1D(int _C,int _K,int _D,
      std::vector<float>&& _W,
      std::vector<float>&& _b)
    : C(_C), K(_K), D(_D),
      bufL((K-1)*D),
      W(std::move(_W)),
      b(std::move(_b)),
      buf(bufL*C, 0.0f),
      head(0)
  {
    assert((int)W.size() == K*C);
    assert((int)b.size() == C);
  }

  // in:  hop×C
  // out: hop×C
  void run(const float* in, float* out, int hop){
    for(int t=0;t<hop;t++){
      const float* inp = in  + t*C;
      float*       outp= out + t*C;
      for(int c=0;c<C;c++){
        float acc = b[c];
        // k=0
        acc += W[c] * inp[c];
        // k=1..K-1
        int idx = head;
        for(int k=1;k<K;k++){
          idx -= D;
          if(idx<0) idx = (idx%bufL + bufL)%bufL;
          acc += W[k*C + c]* buf[idx*C + c];
        }
        outp[c] = acc;
      }
      // write new input into buf
      float* wp = &buf[head*C];
      for(int c=0;c<C;c++) wp[c] = inp[c];
      head = (head+1)%bufL;
    }
  }
};

// ───────────────────────── WEIGHT LOADING ─────────────────────────────
static P1D init_pw, final_pw;
static std::array<D1D,NUM_BLOCKS> sep_conv;
static std::array<P1D,NUM_BLOCKS> pw_sep, pw2;

static P1D loadPW(const std::string &wfile,
                  const std::string &bfile){
  auto Wn = cnpy::npy_load(wfile);
  // shape [1,inC,outC]
  int inC  = (int)Wn.shape[1],
      outC = (int)Wn.shape[2];
  const float* rawW = Wn.data<float>();
  std::vector<float> W(outC*inC);
  // Keras stores [inC,outC], we want W[outC*inC + inC]
  for(int oc=0;oc<outC;oc++)
    for(int ic=0;ic<inC;ic++)
      W[oc*inC + ic] = rawW[ic*outC + oc];
  auto Bn = cnpy::npy_load(bfile);
  const float* rawB = Bn.data<float>();
  std::vector<float> B(rawB, rawB+outC);
  return P1D(inC,outC, std::move(W), std::move(B));
}

static D1D loadDW(const std::string &wfile, int dilation){
  auto Wn = cnpy::npy_load(wfile);
  // shape is [K,C,1]
  int K_ = (int)Wn.shape[0],
      C_ = (int)Wn.shape[1];
  const float* rawW = Wn.data<float>();

  // Keras/TFLite conv1d is actually a *cross-correlation* (no kernel‐flip),
  // but our streaming run() implements the usual convolution
  // (look-back).  So we must *reverse* the kernel now.
  std::vector<float> W(K_*C_);
  for(int k=0; k<K_; k++){
    int kr = (K_-1) - k;           // reverse index
    for(int c=0; c<C_; c++){
      W[k*C_ + c] = rawW[kr*C_ + c];
    }
  }

  // SeparableConv1D’s depthwise part has no bias
  std::vector<float> B(C_, 0.0f);

  return D1D(C_, K_, dilation,
             std::move(W),
             std::move(B));
}

static void loadAll(const std::string &dir){
  init_pw  = loadPW(dir+"/init_pw_w.npy",  dir+"/init_pw_b.npy");
  final_pw = loadPW(dir+"/final_pw_w.npy", dir+"/final_pw_b.npy");
  for(int i=0;i<NUM_BLOCKS;i++){
    sep_conv[i] = loadDW(
      dir+"/dw"+std::to_string(i)+"_w.npy",
      DILATIONS[i]
    );
    pw_sep[i] = loadPW(
      dir+"/pwsep"+std::to_string(i)+"_w.npy",
      dir+"/pwsep"+std::to_string(i)+"_b.npy"
    );
    pw2[i] = loadPW(
      dir+"/pw"+std::to_string(i)+"_w.npy",
      dir+"/pw"+std::to_string(i)+"_b.npy"
    );
  }
}

// ───────────────────────── STREAMING THREAD ──────────────────────────
void inferenceThread(Shared* sh, int hop){

  // 1) Build a little "warm-up" buffer the same size TFLite uses:
  static constexpr int RF = RECEPTIVE_FIELD;                   // 3907
  static constexpr int SEG_LEN = RECEPTIVE_FIELD + MAX_HOP;    // = RF + hop

  // 2) Wait until we've seen a full SEG_LEN samples
  while (!gStopFlag && sh->in.available() < (size_t)SEG_LEN) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (gStopFlag) return;

  // 3) Grab that entire window
  std::vector<float> warm(SEG_LEN);
  sh->in.pop(warm.data(), SEG_LEN);

  // 4) Run it once through your conv-chain *just* to fill the D1D buffers.
  //    You don't need to keep the output.  But you do need to push the
  //    last 'RF' samples back into the ring so that the steady-state
  //    loop picks up the right context.
  //
  //    Here is a quick example:
  {
    // a) feed 'warm' into your init_pw to get buf0:
    std::vector<float> buf0(SEG_LEN*FILTERS);
    init_pw.run(warm.data(), buf0.data(), SEG_LEN);

    // b) run through each sep+pw block *in exactly the same broken-in* way
    //    (copy buf0→res, then loop blocks exactly as below, except:
    //      • do NOT push any output yet
    //      • use the same tmp_pw2 trick above to never overwrite your state)
    //
    std::vector<float> res(SEG_LEN*FILTERS), sep(SEG_LEN*FILTERS), out2(SEG_LEN*FILTERS);

    // copy into res
    std::copy(buf0.begin(), buf0.end(), res.begin());

    for (int b = 0; b < NUM_BLOCKS; b++) {
      sep_conv[b].run(res.data(), sep.data(), SEG_LEN);
      pw_sep[b].run(sep.data(), out2.data(), SEG_LEN);
      // relu
      for (int i = 0; i < SEG_LEN*FILTERS; i++)
        if (out2[i] < 0.0f) out2[i] = 0.0f;

      // safe pw2 into temp
      std::vector<float> tmp_pw2(SEG_LEN*FILTERS);
      pw2[b].run(out2.data(), tmp_pw2.data(), SEG_LEN);

      // residual add
      for (int i = 0; i < SEG_LEN*FILTERS; i++)
        tmp_pw2[i] += res[i];

      // swap
      res.swap(tmp_pw2);
    }

    // final 1×1 + global skip
    std::vector<float> final_out(SEG_LEN);
    final_pw.run(res.data(), final_out.data(), SEG_LEN);
    for (int i = 0; i < SEG_LEN; i++)
      final_out[i] += warm[i];
    // … discard final_out …
  }

  // 5) Now push *only* the LAST 'SEG_LEN - RF' samples back so that
  //    your steady-state loop will see the same buffer that TFLite
  //    sees on its first real Invoke().
  //
  //    SEG_LEN - RF == MAX_HOP == hop, so:
  sh->in.push(warm.data() + RF, hop);

  // 6) Now you can enter your existing hop-by-hop loop below,
  //    and you will *never* under-run and every invocation will
  //    have a full RF of true context behind it.

  // per-hop scratch
  std::vector<float> in_hop(hop),
                     buf0(hop*FILTERS),
                     res(hop*FILTERS),
                     sep(hop*FILTERS),
                     out2(hop*FILTERS),
                     final_out(hop);

  while(!gStopFlag){
    // wait for at least hop samples
    while(!gStopFlag && sh->in.available() < (size_t)hop)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if(gStopFlag) break;

    // read hop
    size_t got = sh->in.pop(in_hop.data(), hop);
    if(got < (size_t)hop)
      std::fill(in_hop.begin()+got, in_hop.end(), 0.0f);

    // 1×1 init
    init_pw.run(in_hop.data(), buf0.data(), hop);

    // 6 residual sep+pw blocks
    std::copy(buf0.begin(), buf0.end(), res.begin());
    for(int b=0;b<NUM_BLOCKS;b++){
      sep_conv[b].run(res.data(), sep.data(), hop);
      pw_sep[b].run(sep.data(), out2.data(), hop);
      // relu
      for(int i=0;i<hop*FILTERS;i++)
        if(out2[i]<0.0f) out2[i]=0.0f;
      
          // … after your pw_sep + ReLU …

      // --- “safe in-place” pw2 via a little temporary ---
      std::vector<float> tmp_pw2(hop*FILTERS);
      pw2[b].run(out2.data(), tmp_pw2.data(), hop);

      // residual add
      for(int i = 0; i < hop*FILTERS; i++)
        tmp_pw2[i] += res[i];

      // now swap into 'res' for the next block
      res.swap(tmp_pw2);
      // reuse out2 as the scratch for the next iteration:
      // (we don't actually need to clear it; it'll be overwritten)
    }

    // final 1×1 + global skip
    final_pw.run(res.data(), final_out.data(), hop);
    for(int i=0;i<hop;i++)
      final_out[i] += in_hop[i];

    sh->out.push(final_out.data(), hop);
  }
}

// ───────────────────────── PA CALLBACK ─────────────────────────────
static int paCallback(const void* inputBuffer,
                      void* outputBuffer,
                      unsigned long framesPerBuffer,
                      const PaStreamCallbackTimeInfo*,
                      PaStreamCallbackFlags,
                      void* userData)
{
  auto* sh = static_cast<Shared*>(userData);
  auto* in = static_cast<const float*>(inputBuffer);
  auto* out= static_cast<float*>(outputBuffer);
  size_t hop = framesPerBuffer;

  // always feed the net
  if(in) sh->in.push(in, hop);

  // bypass?
  if(bypassFlag.load()){
    if(in){
      for(size_t i=0;i<hop;i++){
        float v = in[i];
        out[2*i  ] = v;
        out[2*i+1] = v;
      }
    } else {
      memset(out,0,sizeof(float)*2*hop);
    }
    return paContinue;
  }

  // else normal processed
  float* temp = sh->tempOut.data();
  size_t got  = sh->out.pop(temp, hop);
  if(got < hop)
    memset(temp+got, 0, (hop-got)*sizeof(float));
  for(size_t i=0;i<hop;i++){
    float v = temp[i];
    out[2*i  ] = v;
    out[2*i+1] = v;
  }
  return paContinue;
}

// ───────────────────────── DEVICE LISTING ───────────────────────
static void listDevices(){
  Pa_Initialize();
  int n = Pa_GetDeviceCount();
  std::cout<<"PortAudio devices:\n";
  for(int i=0;i<n;i++){
    const auto* d = Pa_GetDeviceInfo(i);
    std::cout<<" ["<<i<<"] "<<d->name
             <<"  in="<<d->maxInputChannels
             <<"  out="<<d->maxOutputChannels
             <<"  sr="<<d->defaultSampleRate<<"\n";
  }
  Pa_Terminate();
  exit(0);
}

// ───────────────────────── MAIN ─────────────────────────────────
int main(int argc, char** argv){
  int    blockSize = MAX_HOP;
  int    inDev = -1, outDev = -1;
  bool   listDev = false;
  std::string wdir = ".";

  for(int i=1;i<argc;i++){
    if      (!strcmp(argv[i],"--list-devices"))       listDev = true;
    else if (!strcmp(argv[i],"--blocksize") && i+1<argc)
      blockSize = atoi(argv[++i]);
    else if (!strcmp(argv[i],"--in-dev")   && i+1<argc)
      inDev = atoi(argv[++i]);
    else if (!strcmp(argv[i],"--out-dev")  && i+1<argc)
      outDev= atoi(argv[++i]);
    else if (!strcmp(argv[i],"--weights-dir") && i+1<argc)
      wdir  = argv[++i];
    else {
      std::cerr<<"Unknown arg: "<<argv[i]<<"\n"; return 1;
    }
  }

  if(listDev)   listDevices();
  if(inDev<0||outDev<0){
    std::cerr<<"Error: --in-dev N and --out-dev M are required\n";
    listDevices();
  }
  if(blockSize>MAX_HOP){
    std::cerr<<"Error: --blocksize must be ≤ "<<MAX_HOP<<"\n";
    return 1;
  }

  std::cout<<"[MAIN] Weights dir: "<<wdir<<"\n";
  loadAll(wdir);
  std::cout<<"[MAIN] Loaded weights.\n";

  PaError err = Pa_Initialize();
  if(err!=paNoError){
    std::cerr<<"Pa_Initialize failed: "<<Pa_GetErrorText(err)<<"\n";
    return 1;
  }

  const PaDeviceInfo* diIn  = Pa_GetDeviceInfo(inDev);
  const PaDeviceInfo* diOut = Pa_GetDeviceInfo(outDev);
  if(!diIn||!diOut){
    std::cerr<<"Bad device index\n"; Pa_Terminate(); return 1;
  }
  std::cout<<"[MAIN] Input  Dev ["<<inDev<<"] "<<diIn->name
           <<"  inCh="<<diIn->maxInputChannels
           <<"  sr="<<diIn->defaultSampleRate<<"\n";
  std::cout<<"[MAIN] Output Dev ["<<outDev<<"] "<<diOut->name
           <<"  outCh="<<diOut->maxOutputChannels
           <<"  sr="<<diOut->defaultSampleRate<<"\n";

  int sampleRate = (int)std::lround(diIn->defaultSampleRate);
  std::cout<<"[MAIN] blockSize="<<blockSize
           <<"  RECEPTIVE_FIELD="<<RECEPTIVE_FIELD<<"\n";

  PaStreamParameters ip = { inDev, 1, paFloat32,
    diIn->defaultLowInputLatency, nullptr };
  PaStreamParameters op = { outDev,2, paFloat32,
    diOut->defaultLowOutputLatency,nullptr };

  // ring capacity ≥ RECEPTIVE_FIELD + hop
  size_t cap=1;
  while(cap < (size_t)(RECEPTIVE_FIELD + blockSize)) cap <<= 1;
  std::cout<<"[MAIN] RingBuffer cap="<<cap<<" samples\n";

  Shared shared(cap);
  shared.tempOut.resize(blockSize);

  // switch stdin → nonblocking raw mode
  struct termios oldt, newt;
  tcgetattr(STDIN_FILENO,&oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON|ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);

  std::signal(SIGINT, signalHandler);

  // spawn inference thread
  std::thread infThread([&](){
    inferenceThread(&shared, blockSize);
  });

  // spawn keyboard thread for A/B toggle
  std::thread kbThread([&](){
    while(!gStopFlag){
      int c = getchar();
      if(c=='b'||c=='B'){
        bool now = !bypassFlag.load();
        bypassFlag.store(now);
        // flush stale model-output
        shared.out.readIdx.store(
          shared.out.writeIdx.load(std::memory_order_relaxed),
          std::memory_order_relaxed
        );
        std::cout<<"[MAIN] Bypass "<<(now?"ON":"OFF")<<"\n";
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });

  PaStream* stream=nullptr;
  err = Pa_OpenStream(&stream,
                      &ip,&op,
                      sampleRate,
                      blockSize,
                      paNoFlag,
                      paCallback,
                      &shared);
  if(err!=paNoError){
    std::cerr<<"Pa_OpenStream failed: "<<Pa_GetErrorText(err)<<"\n";
    Pa_Terminate(); return 1;
  }
  err = Pa_StartStream(stream);
  if(err!=paNoError){
    std::cerr<<"Pa_StartStream failed: "<<Pa_GetErrorText(err)<<"\n";
    Pa_CloseStream(stream);
    Pa_Terminate();
    return 1;
  }

  std::cout<<"[MAIN] Streaming...  Press 'b' to A/B, Ctrl-C to quit.\n";
  while(!gStopFlag) Pa_Sleep(100);

  std::cout<<"[MAIN] Stopping...\n";
  Pa_StopStream(stream);
  Pa_CloseStream(stream);
  Pa_Terminate();

  infThread.join();
  kbThread.join();

  // restore stdin
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);

  std::cout<<"[MAIN] Done.\n";
  return 0;
}