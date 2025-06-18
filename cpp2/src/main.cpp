// stream_piezo2mic_ab.cpp

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
#include <portaudio.h>
#include <stdexcept>
#include "cnpy.h"
#include <Accelerate/Accelerate.h>  // For cblas_sgemm

// ───────────────────────── PARAMETERS ─────────────────────────
static constexpr int FILTERS   = 32;
static constexpr int KERNEL    = 31;
static constexpr std::array<int,6> DILATIONS = {1,2,4,8,16,32};
static constexpr int NUM_BLOCKS = (int)DILATIONS.size();
static constexpr int MAX_HOP    = 64;
static constexpr int RECEPTIVE_FIELD =
  1 + (KERNEL - 1) * (1+2+4+8+16+32);

// ───────────────────────── GLOBAL FLAGS ───────────────────────
std::atomic<bool> gStopFlag(false);

std::atomic<bool> bypassFlag(false);
bool             gUseInt8   = false;
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
struct Shared { RingBuffer in, out; std::vector<float> tempOut; Shared(size_t cap): in(cap), out(cap){} };

// ───────────────────────── STREAMING CONV KERNELS ─────────────────────
// 1×1 pointwise conv
struct P1D {
  int inC, outC;
  std::vector<float> W;    // W: inC*outC row-major
  std::vector<float> b;

  P1D(){}
  P1D(int _inC,int _outC, std::vector<float>&& _W, std::vector<float>&& _b)
    : inC(_inC), outC(_outC), W(std::move(_W)), b(std::move(_b)){}

  void run(const float* in, float* out, int hop) const {
    // C = A*B + broadcast(b)
    // A: hop x inC, B: inC x outC, C: hop x outC
    cblas_sgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      hop, outC, inC,
      1.f,
      in, inC,
      W.data(), outC,
      0.f,
      out, outC
    );
    // add bias
    for(int t=0;t<hop;++t){
      float* row = out + t*outC;
      for(int oc=0;oc<outC;++oc)
        row[oc] += b[oc];
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

static P1D loadPW(const std::string &wfile, const std::string &bfile){
  auto Wn = cnpy::npy_load(wfile);
  int inC = Wn.shape[1], outC = Wn.shape[2];

  const float* rawW = Wn.data<float>();
  // store as inC*outC row-major
  std::vector<float> W(inC*outC);
  for(int ic=0;ic<inC;++ic)
    for(int oc=0;oc<outC;++oc)
      W[ic*outC + oc] = rawW[ic*outC + oc];
  auto Bn = cnpy::npy_load(bfile);
  std::vector<float> B(Bn.data<float>(), Bn.data<float>()+outC);
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

// ----------------------------------------------------------------
// helper: parse a two-column "scale,zero_point" text file
// ----------------------------------------------------------------
static void parseQuantFile(const std::string &fname,
                           std::vector<float> &scales,
                           std::vector<int>   &zps)
{
  std::ifstream ifs(fname);
  if(!ifs){ std::cerr<<"ERROR: can't open "<<fname<<"\n"; std::exit(1);}
  std::string line;
  // skip header
  std::getline(ifs,line);
  while(std::getline(ifs,line)){
    if(line.empty()) continue;
    std::stringstream ss(line);
    std::string s0,s1;
    std::getline(ss,s0,',');
    std::getline(ss,s1,',');
    scales.push_back(std::stof(s0));
    zps   .push_back(std::stoi(s1));
  }
}

static void loadAllFp32(const std::string &dir){
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

// ───────────────────────── INT8→FLOAT WEIGHT LOADER ──────────────────────
static P1D loadPWQuant(const std::string &base) {
  auto Wn = cnpy::npy_load(base + "_w_int8.npy");
  auto Bn = cnpy::npy_load(base + "_b_int32.npy");

  std::vector<float> wscale;
  std::vector<int>   wzp;
  std::vector<float> bscale;
  std::vector<int>   bzp;
  parseQuantFile(base + "_w_int8_quant.txt", wscale, wzp);
  parseQuantFile(base + "_b_int32_quant.txt", bscale, bzp);

  // TFLite packs 1×1 conv in [outC,1,1,inC]
  const auto &shape = Wn.shape;
  int outC, inC;
  if (shape.size() == 4 && shape[1] == 1 && shape[2] == 1) {
    outC = int(shape[0]);
    inC  = int(shape[3]);
  } else if (shape.size() == 2) {
    // if you pre-flattened to [inC,outC]
    inC  = int(shape[0]);
    outC = int(shape[1]);
  } else {
    throw std::runtime_error("loadPWQuant: unexpected weight shape");
  }

  size_t N       = size_t(inC) * size_t(outC);
  auto   rawW8   = Wn.data<int8_t>();
  std::vector<float> W(N);

  // Dequant + reshape to (inC × outC)
  if (wscale.size() == 1) {
    float S = wscale[0]; int Z = wzp[0];
    for (int oc = 0; oc < outC; ++oc) {
      for (int ic = 0; ic < inC; ++ic) {
        int raw_idx = oc*inC + ic;        // NHWC layout
        int dst_idx = ic*outC + oc;       // row-major
        W[dst_idx] = (float(rawW8[raw_idx]) - Z)*S;
      }
    }
  } else {
    for (int oc = 0; oc < outC; ++oc) {
      float S = wscale[oc]; int Z = wzp[oc];
      for (int ic = 0; ic < inC; ++ic) {
        int raw_idx = oc*inC + ic;
        int dst_idx = ic*outC + oc;
        W[dst_idx] = (float(rawW8[raw_idx]) - Z)*S;
      }
    }
  }

  // Dequant bias
  auto rawB32 = Bn.data<int32_t>();
  std::vector<float> B(outC);
  if (bscale.size() == 1) {
    float S = bscale[0]; int Z = bzp[0];
    for (int oc = 0; oc < outC; ++oc) {
      B[oc] = (float(rawB32[oc]) - Z)*S;
    }
  } else {
    for (int oc = 0; oc < outC; ++oc) {
      B[oc] = (float(rawB32[oc]) - bzp[oc])*bscale[oc];
    }
  }

  return P1D(inC, outC, std::move(W), std::move(B));
}


static D1D loadDWQuant(const std::string &wf,
                       const std::string &qf,
                       int dilation)
{
  auto Wn = cnpy::npy_load(wf);
  std::vector<float> scale;
  std::vector<int>   zp;
  parseQuantFile(qf, scale, zp);

  // TFLite depthwise is [1,1,K,inC]
  const auto &shape = Wn.shape;
  int K, C;
  if (shape.size() == 4 && shape[0] == 1 && shape[1] == 1) {
    K = int(shape[2]);
    C = int(shape[3]);
  } else if (shape.size() == 2) {
    // if you flattened to [K,C]
    K = int(shape[0]);
    C = int(shape[1]);
  } else {
    throw std::runtime_error("loadDWQuant: unexpected weight shape");
  }

  size_t N        = size_t(K)*size_t(C);
  auto rawW8      = Wn.data<int8_t>();
  std::vector<float> W(N);

  // Dequant + conv-flip
  if (scale.size() == 1) {
    float S = scale[0]; int Z = zp[0];
    for (int k = 0; k < K; ++k) {
      for (int c = 0; c < C; ++c) {
        int raw_idx = k*C + c;
        int dst_idx = (K-1-k)*C + c;
        W[dst_idx] = (float(rawW8[raw_idx]) - Z)*S;
      }
    }
  } else {
    for (int c = 0; c < C; ++c) {
      float S = scale[c]; int Z = zp[c];
      for (int k = 0; k < K; ++k) {
        int raw_idx = k*C + c;
        int dst_idx = (K-1-k)*C + c;
        W[dst_idx] = (float(rawW8[raw_idx]) - Z)*S;
      }
    }
  }

  // no bias in depthwise
  std::vector<float> B(C, 0.0f);
  return D1D(C, K, dilation, std::move(W), std::move(B));
}

static void loadAllInt8(const std::string &dir){
  init_pw  = loadPWQuant(dir+"/init_pw");
  final_pw = loadPWQuant(dir+"/final_pw");
  for(int i=0;i<NUM_BLOCKS;i++){
    sep_conv[i] = loadDWQuant(
                     dir+"/dw"+std::to_string(i)+"_w_int8.npy",
                     dir+"/dw"+std::to_string(i)+"_w_int8_quant.txt",
                     DILATIONS[i]
                   );
    pw_sep[i] = loadPWQuant(dir+"/pwsep"+std::to_string(i));
    pw2  [i] = loadPWQuant(dir+"/pw"   +std::to_string(i));
  }
}

// ───────────────────────── STREAMING THREAD ──────────────────────────
void inferenceThread(Shared* sh, int hop) {
  // 1) Compute receptive‐field (RF) and segment‐length = RF + hop
  const int RF      = RECEPTIVE_FIELD;  
  const int SEG_LEN = RF + hop;

  // 2) One‐time “warm‐up”: wait until we have SEG_LEN samples in
  while (!gStopFlag && sh->in.available() < (size_t)SEG_LEN) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (gStopFlag) return;

  // 3) Pop them all into a warm buffer
  std::vector<float> warm(SEG_LEN);
  sh->in.pop(warm.data(), SEG_LEN);

  // 4) Run that entire window through the conv‐chain JUST to fill
  //    your D1D buffers.  We discard the output, but keep state.
  {
    // a) initial 1×1
    std::vector<float> buf0(SEG_LEN * FILTERS);
    init_pw.run(warm.data(), buf0.data(), SEG_LEN);

    // b) each separable‐res block
    std::vector<float> res(buf0), sep(SEG_LEN*FILTERS), out2(SEG_LEN*FILTERS);
    for (int b = 0; b < NUM_BLOCKS; ++b) {
      sep_conv[b].run(res.data(), sep.data(), SEG_LEN);
      pw_sep[b].run(sep.data(), out2.data(), SEG_LEN);
      // ReLU
      for (int i = 0; i < SEG_LEN*FILTERS; ++i)
        if (out2[i] < 0.0f) out2[i] = 0.0f;
      // safe‐in‐place 1×1 → tmp
      std::vector<float> tmp_pw2(SEG_LEN*FILTERS);
      pw2[b].run(out2.data(), tmp_pw2.data(), SEG_LEN);
      // residual add
      for (int i = 0; i < SEG_LEN*FILTERS; ++i)
        tmp_pw2[i] += res[i];
      res.swap(tmp_pw2);
    }

    // c) final 1×1 + global skip
    std::vector<float> final_out(SEG_LEN);
    final_pw.run(res.data(), final_out.data(), SEG_LEN);
    for (int i = 0; i < SEG_LEN; ++i)
      final_out[i] += warm[i];
    // → drop final_out
  }

  // 5) Now push back *only* the last hop samples so that the
  //    steady‐state loop has the correct tail‐context
  sh->in.push(warm.data() + RF, hop);

  // 6) Steady‐state: process hop‐sized blocks forever
  std::vector<float> in_hop(hop),
                     buf0(hop*FILTERS),
                     res(hop*FILTERS),
                     sep(hop*FILTERS),
                     out2(hop*FILTERS),
                     final_out(hop);

  while (!gStopFlag) {
    // a) wait for hop samples
    while (!gStopFlag && sh->in.available() < (size_t)hop)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (gStopFlag) break;

    // b) pop
    size_t got = sh->in.pop(in_hop.data(), hop);
    if (got < (size_t)hop)
      std::fill(in_hop.begin()+got, in_hop.end(), 0.0f);

    // c) init 1×1
    init_pw.run(in_hop.data(), buf0.data(), hop);

    // d) sep‐res blocks
    std::copy(buf0.begin(), buf0.end(), res.begin());
    for (int b = 0; b < NUM_BLOCKS; ++b) {
      sep_conv[b].run(res.data(), sep.data(), hop);
      pw_sep[b].run(sep.data(), out2.data(), hop);
      for (int i = 0; i < hop*FILTERS; ++i)
        if (out2[i] < 0.0f) out2[i] = 0.0f;
      // safe pw2
      std::vector<float> tmp_pw2(hop*FILTERS);
      pw2[b].run(out2.data(), tmp_pw2.data(), hop);
      for (int i = 0; i < hop*FILTERS; ++i)
        tmp_pw2[i] += res[i];
      res.swap(tmp_pw2);
    }

    // e) final 1×1 + skip
    final_pw.run(res.data(), final_out.data(), hop);
    for (int i = 0; i < hop; ++i)
      final_out[i] += in_hop[i];

    // f) push out
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
    else if (!strcmp(argv[i],"--int8")) {
      gUseInt8 = true;
    }
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

  // load either FP32 or INT8→float
  std::cout<<"[MAIN] Weights dir: "<<wdir<<"\n";
  std::cout<<"[MAIN] Model mode: "<<(gUseInt8?"INT8":"FP32")<<"\n";
  if (gUseInt8){
    loadAllInt8(wdir);
  } else{
    loadAllFp32(wdir);
  }
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

  // compute the raw minimum capacity: RF + hop
  size_t needed = RECEPTIVE_FIELD + blockSize;

  // then pick the smallest power‑of‑two ≥ 2×needed
  size_t cap = 1;
  while (cap < 2 * needed) cap <<= 1;

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