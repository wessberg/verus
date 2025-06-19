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
#include <portaudio.h>
#include "cnpy.h"
#include <pthread/qos.h>
#include <Accelerate/Accelerate.h> // for cblas_sgemm

// ───────────────────────── PARAMETERS ─────────────────────────
static constexpr int FILTERS        = 64;
static constexpr int KERNEL         = 51;
static constexpr std::array<int,6> DILATIONS     = {1,2,4,8,16,32};
static constexpr int NUM_BLOCKS     = (int)DILATIONS.size();
static constexpr int MAX_HOP        = 64;
static constexpr int RECEPTIVE_FIELD=
    1 + (KERNEL-1)*(1+2+4+8+16+32);

// ───────────────────────── GLOBAL FLAGS ───────────────────────
std::atomic<bool> gStopFlag(false), bypassFlag(false);
bool gUseInt8 = false;
void signalHandler(int){ gStopFlag = true; }

// ───────────────────────── RING BUFFER ────────────────────────
struct RingBuffer {
  std::vector<float> buf;
  size_t mask;
  std::atomic<size_t> r{0}, w{0};
  RingBuffer(size_t cap): buf(cap), mask(cap-1){
    assert((cap & mask)==0); // must be power-of-two
  }
  size_t available() const {
    return (w.load(std::memory_order_acquire)
          - r.load(std::memory_order_acquire)) & mask;
  }
  size_t freeSpace() const {
    return (mask+1) - available();
  }
  size_t push(const float *data, size_t n){
    size_t free = freeSpace(), toW = std::min(n, free);
    size_t w0 = w.load(std::memory_order_relaxed);
    for(size_t i=0;i<toW;i++) buf[(w0+i)&mask] = data[i];
    w.store((w0+toW)&mask, std::memory_order_release);
    return toW;
  }
  size_t pop(float *out, size_t n){
    size_t av = available(), toR = std::min(n,av);
    size_t r0 = r.load(std::memory_order_relaxed);
    for(size_t i=0;i<toR;i++) out[i] = buf[(r0+i)&mask];
    r.store((r0+toR)&mask, std::memory_order_release);
    return toR;
  }
};

struct Shared {
  RingBuffer in, out;
  std::vector<float> tempOut;
  Shared(size_t cap): in(cap), out(cap){}
};

// ────────────────────────────── CONV LAYERS ────────────────────────
// 1×1 pointwise conv
struct P1D {
  int inC, outC;
  std::vector<float> W, b;
  P1D(): inC(0), outC(0) {}
  P1D(int _inC,int _outC,
      std::vector<float>&&_W,
      std::vector<float>&&_b)
    : inC(_inC), outC(_outC),
      W(std::move(_W)),
      b(std::move(_b))
  {}
  void run(const float *in, float *out, int hop) const {
    cblas_sgemm(
      CblasRowMajor,
      CblasNoTrans, CblasNoTrans,
      hop, outC, inC,
      1.f,
      in, inC,
      W.data(), outC,
      0.f,
      out, outC
    );
    for(int t=0;t<hop;t++){
      float *row = out + t*outC;
      for(int oc=0;oc<outC;oc++)
        row[oc] += b[oc];
    }
  }
};

// Depthwise dilated conv (causal)
struct D1D {
  int C,K,D,bufL;
  std::vector<float> W,b,buf;
  int head=0;
  D1D():C(0),K(0),D(0),bufL(0){}
  D1D(int _C,int _K,int _D,
      std::vector<float>&&_W,
      std::vector<float>&&_b)
    : C(_C), K(_K), D(_D),
      bufL((K-1)*D),
      W(std::move(_W)),
      b(std::move(_b)),
      buf(bufL*C,0.0f),
      head(0)
  {
    assert((int)W.size()==K*C);
    assert((int)b.size()==C);
  }
  void run(const float *in, float *out, int hop){
    for(int t=0;t<hop;t++){
      const float *inp = in + t*C;
      float *outp      = out + t*C;
      for(int c=0;c<C;c++){
        float acc = b[c];
        // k=0
        acc += W[c] * inp[c];
        // k=1..K-1
        int idx = head;
        for(int k=1;k<K;k++){
          idx -= D;
          if(idx < 0) idx = (idx % bufL + bufL) % bufL;
          acc += W[k*C + c] * buf[idx*C + c];
        }
        outp[c] = acc;
      }
      // push into circular buf
      float *wp = &buf[head*C];
      for(int c=0;c<C;c++) wp[c] = inp[c];
      head = (head+1) % bufL;
    }
  }
};

// ─────────────────────────── UTILITY ────────────────────────────
static inline bool file_exists(const std::string &fn){
  struct stat st;
  return stat(fn.c_str(), &st)==0;
}

// parse two-column "scale zero_point" text file
static void parseQuantFile(const std::string &fname,
                           std::vector<float> &scales,
                           std::vector<int>   &zps)
{
  std::ifstream ifs(fname);
  if(!ifs){
    std::cerr<<"ERROR: cannot open "<<fname<<"\n";
    std::exit(1);
  }
  std::string line;
  std::getline(ifs,line); // skip header
  while(std::getline(ifs,line)){
    if(line.empty()) continue;
    for(char &c:line) if(c==',') c=' ';
    std::istringstream ss(line);
    float s; int zp;
    ss>>s>>zp;
    if(!ss) throw std::runtime_error("bad quant line: "+line);
    scales.push_back(s);
    zps.push_back(zp);
  }
}

// ───────────────────────── FP32 WEIGHT LOADERS ─────────────────────
static P1D loadPW(const std::string &wfile,
                  const std::string &bfile)
{
  auto Wn = cnpy::npy_load(wfile);
  // Wn.shape = [kernel, inC, outC]
  int inC  = (int)Wn.shape[1],
      outC = (int)Wn.shape[2];
  const float *rawW = Wn.data<float>();
  std::vector<float> W(inC*outC);
  // flatten kernel=1
  for(int ic=0;ic<inC;ic++)
    for(int oc=0;oc<outC;oc++)
      W[ic*outC + oc] = rawW[ic*outC + oc];
  auto Bn = cnpy::npy_load(bfile);
  std::vector<float> B(outC);
  std::copy(Bn.data<float>(),
            Bn.data<float>()+outC,
            B.begin());
  return P1D(inC,outC,std::move(W),std::move(B));
}

static D1D loadDW(const std::string &wfile,
                  int dilation)
{
  auto Wn = cnpy::npy_load(wfile);
  // Wn.shape = [K, C]
  int K = (int)Wn.shape[0],
      C = (int)Wn.shape[1];
  const float *rawW = Wn.data<float>();
  // reverse kernel order for convolution
  std::vector<float> W((size_t)K*C);
  for(int k=0;k<K;k++){
    int kr = (K-1)-k;
    for(int c=0;c<C;c++){
      W[k*C + c] = rawW[kr*C + c];
    }
  }
  // no bias
  std::vector<float> B(C, 0.0f);
  return D1D(C,K,dilation,std::move(W),std::move(B));
}

// ─────────────────────────── INT8 WEIGHT LOADERS ───────────────────
// load pointwise (1×1) quant layer; expects EXACT inC/outC
static P1D loadPWQuant(const std::string &base,
                       int exp_inC,
                       int exp_outC)
{
  auto Wn = cnpy::npy_load(base + "_w_int8.npy");
  std::cerr<<"[loadPWQuant] "<<base<<" weight shape=(";
  for(auto &d:Wn.shape) std::cerr<<d<<","; std::cerr<<")\n";
  auto Bn = cnpy::npy_load(base + "_b_int32.npy");
  std::cerr<<"[loadPWQuant] "<<base<<" bias   shape=(";
  for(auto &d:Bn.shape) std::cerr<<d<<","; std::cerr<<")\n";

  // parse quant params
  std::vector<float> wscale; std::vector<int> wzp;
  std::vector<float> bscale; std::vector<int> bzp;
  parseQuantFile(base + "_w_int8_quant.txt", wscale, wzp);
  parseQuantFile(base + "_b_int32_quant.txt", bscale, bzp);

  // detect layout
  const auto &sh = Wn.shape;
  int inC=0,outC=0;
  enum L{HWIO,OIHW,FLAT} layout=FLAT;
  bool ok=false;
  if(sh.size()==4){
    int d0=sh[0], d1=sh[1], d2=sh[2], d3=sh[3];
    // TF-Lite: [1,1,inC,outC]
    if(d0==1 && d1==1 && d2==exp_inC && d3==exp_outC){
      inC=d2; outC=d3; layout=HWIO; ok=true;
    }
    // old: [outC,1,1,inC]
    else if(d0==exp_outC && d1==1 && d2==1 && d3==exp_inC){
      inC=d3; outC=d0; layout=OIHW; ok=true;
    }
    // flat in 4d: [inC,outC,1,1]
    else if(d0==exp_inC && d1==exp_outC && d2==1 && d3==1){
      inC=d0; outC=d1; layout=FLAT; ok=true;
    }
  }
  else if(sh.size()==2){
    int d0=sh[0], d1=sh[1];
    if(d0==exp_inC && d1==exp_outC){
      inC=d0; outC=d1; layout=FLAT; ok=true;
    }
  }
  if(!ok){
    std::cerr<<"[ERROR] loadPWQuant("<<base<<") expected (inC,outC)=("
             <<exp_inC<<","<<exp_outC<<") but got shape(";
    for(auto &d:sh) std::cerr<<d<<","; std::cerr<<")\n";
    std::exit(1);
  }

  // dequant + reshape
  auto rawW8 = Wn.data<int8_t>();
  size_t N = (size_t)inC*(size_t)outC;
  std::vector<float> W(N);
  auto getS=[&](auto &V,int oc){ return V.size()==1?V[0]:V[oc]; };
  auto getZ=[&](auto &V,int oc){ return V.size()==1?V[0]:V[oc]; };

  if(layout==HWIO){
    for(int ic=0;ic<inC;ic++)
      for(int oc=0;oc<outC;oc++){
        int idx = ic*outC + oc;
        float S = getS(wscale,oc);
        int   Z = getZ(wzp,   oc);
        W[idx] = (rawW8[idx] - Z)*S;
      }
  } else if(layout==OIHW){
    for(int oc=0;oc<outC;oc++)
      for(int ic=0;ic<inC;ic++){
        int raw = oc*inC + ic;
        int dst = ic*outC + oc;
        float S = getS(wscale,oc);
        int   Z = getZ(wzp,   oc);
        W[dst] = (rawW8[raw] - Z)*S;
      }
  } else {
    for(int ic=0;ic<inC;ic++)
      for(int oc=0;oc<outC;oc++){
        int idx = ic*outC + oc;
        float S = getS(wscale,oc);
        int   Z = getZ(wzp,   oc);
        W[idx] = (rawW8[idx] - Z)*S;
      }
  }

  // bias
  auto rawB = Bn.data<int32_t>();
  std::vector<float> B(outC);
  for(int oc=0;oc<outC;oc++){
    float S = getS(bscale,oc);
    int   Z = getZ(bzp,   oc);
    B[oc] = (rawB[oc] - Z)*S;
  }

  return P1D(inC,outC,std::move(W),std::move(B));
}

// load depthwise quant; expects EXACT K,C
static D1D loadDWQuant(const std::string &wf,
                       const std::string &qf,
                       int dilation,
                       int exp_K,
                       int exp_C)
{
  auto Wn = cnpy::npy_load(wf);
  std::cerr<<"[loadDWQuant] "<<wf<<" shape=(";
  for(auto &d:Wn.shape) std::cerr<<d<<","; std::cerr<<")\n";
  std::vector<float> scale; std::vector<int> zp;
  parseQuantFile(qf, scale, zp);

  const auto &sh = Wn.shape;
  int Kc=0, Cc=0;
  bool ok=false;
  if(sh.size()==4){
    int d0=sh[0], d1=sh[1], d2=sh[2], d3=sh[3];
    // [1,1,K,C]
    if(d0==1 && d1==1 && d2==exp_K && d3==exp_C){
      Kc=d2; Cc=d3; ok=true;
    }
    // [1,K,C,1]
    else if(d0==1 && d1==exp_K && d2==exp_C && d3==1){
      Kc=d1; Cc=d2; ok=true;
    }
    // [K,C,1,1]
    else if(d0==exp_K && d1==exp_C && d2==1 && d3==1){
      Kc=d0; Cc=d1; ok=true;
    }
  }
  else if(sh.size()==2){
    if(sh[0]==exp_K && sh[1]==exp_C){
      Kc=sh[0]; Cc=sh[1]; ok=true;
    }
  }
  if(!ok){
    std::cerr<<"[ERROR] loadDWQuant("<<wf<<") expected (K,C)=("
             <<exp_K<<","<<exp_C<<") but got shape(";
    for(auto &d:sh) std::cerr<<d<<","; std::cerr<<")\n";
    std::exit(1);
  }

  // dequant + flip
  auto rawW8 = Wn.data<int8_t>();
  std::vector<float> W((size_t)Kc*Cc);
  auto getS=[&](int c){ return scale.size()==1?scale[0]:scale[c]; };
  auto getZ=[&](int c){ return zp.size()==1?zp[0]:zp[c]; };
  for(int k=0;k<Kc;k++){
    for(int c=0;c<Cc;c++){
      int raw = k*Cc + c;
      int dst = (Kc-1-k)*Cc + c;
      W[dst] = (rawW8[raw] - getZ(c)) * getS(c);
    }
  }
  std::vector<float> B(Cc, 0.0f);
  return D1D(Cc,Kc,dilation,std::move(W),std::move(B));
}

// ──────────────────────────── LOAD ALL WEIGHTS ─────────────────────
static P1D init_pw, final_pw, skip_pw;
static D1D hf_conv;
static std::array<D1D,NUM_BLOCKS> sep_conv;
static std::array<P1D,NUM_BLOCKS> pw_sep, pw2;

static void loadAllFp32(const std::string &dir){
  // init + final
  init_pw  = loadPW(dir+"/init_pw_w.npy",  dir+"/init_pw_b.npy");
  final_pw = loadPW(dir+"/final_pw_w.npy", dir+"/final_pw_b.npy");
  // gated blocks
  for(int i=0;i<NUM_BLOCKS;i++){
    sep_conv[i] = loadDW(dir+"/dw"+std::to_string(i)+"_w.npy",
                         DILATIONS[i]);
    pw_sep[i]   = loadPW(dir+"/pwsep"+std::to_string(i)+"_w.npy",
                         dir+"/pwsep"+std::to_string(i)+"_b.npy");
    pw2[i]      = loadPW(dir+"/pw"+std::to_string(i)+"_w.npy",
                         dir+"/pw"+std::to_string(i)+"_b.npy");
  }
  // skip_conv (no bias)
  {
    std::string fn = dir + "/skip_conv_w.npy";
    if(!file_exists(fn)){
      std::cerr<<"[loadAllFp32] missing "<<fn<<", making skip=0\n";
      skip_pw = P1D(1,1,std::vector<float>{0.0f},std::vector<float>{0.0f});
    } else {
      auto Wn = cnpy::npy_load(fn);
      // Wn.shape=[1,inC,outC]
      int inC  = (int)Wn.shape[1],
          outC = (int)Wn.shape[2];
      const float *r = Wn.data<float>();
      std::vector<float> W((size_t)inC*outC);
      for(int ic=0;ic<inC;ic++)
        for(int oc=0;oc<outC;oc++)
          W[ic*outC+oc] = r[ic*outC+oc];
      skip_pw = P1D(inC,outC,std::move(W),
                    std::vector<float>(outC,0.0f));
    }
  }
  // hf_transient (3-tap)
  {
    std::string fn = dir + "/hf_transient_w.npy";
    if(!file_exists(fn)){
      std::cerr<<"[loadAllFp32] missing "<<fn<<", HF=0\n";
      hf_conv = D1D(1,1,1,std::vector<float>{0.0f},
                         std::vector<float>{0.0f});
    } else {
      // saved as depthwise [K,C]
      hf_conv = loadDW(fn, 1);
    }
  }
}

static void loadAllInt8(const std::string &dir){
  // init + final
  init_pw  = loadPWQuant(dir+"/init_pw",       1, FILTERS);
  final_pw = loadPWQuant(dir+"/final_pw", FILTERS, 1);
  // gated blocks
  for(int i=0;i<NUM_BLOCKS;i++){
    sep_conv[i] = loadDWQuant(
      dir+"/dw"+std::to_string(i)+"_w_int8.npy",
      dir+"/dw"+std::to_string(i)+"_w_int8_quant.txt",
      DILATIONS[i],
      KERNEL, FILTERS
    );
    pw_sep[i] = loadPWQuant(
      dir+"/pwsep"+std::to_string(i),
      FILTERS, 2*FILTERS
    );
    pw2[i]    = loadPWQuant(
      dir+"/pw"+std::to_string(i),
      FILTERS, FILTERS
    );
  }
  // skip_conv
  skip_pw = loadPWQuant(dir+"/skip_conv", 1, 1);
  // hf_transient (3-tap conv)
  hf_conv = loadDWQuant(
    dir+"/hf_transient_w_int8.npy",
    dir+"/hf_transient_w_int8_quant.txt",
    1, // dilation
    3, // kernel
    1  // channels
  );
}

// ───────────────────────── STREAMING THREAD ─────────────────────────
void inferenceThread(Shared *sh, int hop){
    // (7) raise to USER_INTERACTIVE on macOS
    #if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    #endif

    const int RF      = RECEPTIVE_FIELD;
    const int SEG    = RF + hop;
    const int F      = FILTERS;
    const int OC     = 2*F;

    // — pre-allocated “real-time” buffers —
    std::vector<float> warm(SEG),
                       buf0(SEG*F),
                       res (SEG*F),
                       sep (SEG*F),
                       tmp (SEG*F),
                       pwsep(SEG*OC),
                       gated(SEG*F),
                       finalW(SEG),
                       skipW(SEG),
                       hfW  (SEG);

    std::vector<float> in_hop(hop),
                       buf1  (hop*F),
                       res1  (hop*F),
                       sep1  (hop*F),
                       tmp1  (hop*F),
                       gated1(hop*F),
                       finalH(hop),
                       skipH (hop),
                       hfH   (hop);

    // — warm-up: spin until we have SEG frames in the ring —
    while(!gStopFlag && sh->in.available() < (size_t)SEG)
        std::this_thread::yield();
    if(gStopFlag) return;
    sh->in.pop(warm.data(), SEG);

    // a) initial 1×1
    init_pw.run(warm.data(), buf0.data(), SEG);
    std::copy(buf0.begin(), buf0.end(), res.begin());

    // b) gated dilated blocks
    for(int b = 0; b < NUM_BLOCKS; ++b){
        sep_conv[b].run(res.data(), sep.data(), SEG);
        pw_sep[b].run(sep.data(), pwsep.data(), SEG);

        // gating non-linearity (scalar)
        for(int t = 0; t < SEG; ++t){
            float *s = pwsep.data() + t*OC;
            float *g = gated.data() + t*F;
            for(int c=0; c<F; ++c){
                float x = s[c], y = s[c+F];
                g[c] = std::tanh(x) * (1.f/(1.f + std::exp(-y)));
            }
        }

        pw2[b].run(gated.data(), tmp.data(), SEG);

        // residual
        for(int i=0; i < SEG*F; ++i)
            tmp[i] += res[i];
        res.swap(tmp);
    }

    // c) final 1×1
    final_pw.run(res.data(), finalW.data(), SEG);
    // d) skip & hf
    skip_pw.run(warm.data(), skipW.data(), SEG);
    hf_conv.run(warm.data(),    hfW.data(), SEG);

    // e) merge + tanh (vectorized)
    // finalW += skipW; finalW += hfW; finalW += warm; finalW = tanh(finalW);
    vDSP_vadd(finalW.data(),1, skipW.data(),1, finalW.data(),1, SEG);
    vDSP_vadd(finalW.data(),1, hfW  .data(),1, finalW.data(),1, SEG);
    vDSP_vadd(finalW.data(),1, warm .data(),1, finalW.data(),1, SEG);
    int N = SEG;
    vvtanhf(finalW.data(), finalW.data(), &N);

    // f) push overlap frames back
    sh->in.push(warm.data()+RF, hop);

    // — steady-state loop —
    while(!gStopFlag){
        // (4) spin-wait for next hop
        while(!gStopFlag && sh->in.available() < (size_t)hop)
            std::this_thread::yield();
        if(gStopFlag) break;

        size_t got = sh->in.pop(in_hop.data(), hop);
        if(got < (size_t)hop)
            std::fill(in_hop.begin()+got, in_hop.end(), 0.f);

        // a) init
        init_pw.run(in_hop.data(), buf1.data(), hop);
        std::copy(buf1.begin(), buf1.end(), res1.begin());

        // b) gated blocks
        for(int b=0; b<NUM_BLOCKS; ++b){
            sep_conv[b].run(res1.data(), sep1.data(), hop);
            pw_sep[b].run(sep1.data(), pwsep.data(), hop);

            for(int t=0; t<hop; ++t){
                float *s = pwsep.data() + t*OC;
                float *g = gated1.data()+ t*F;
                for(int c=0; c<F; ++c){
                    float x = s[c], y = s[c+F];
                    g[c] = std::tanh(x) * (1.f/(1.f + std::exp(-y)));
                }
            }

            pw2[b].run(gated1.data(), tmp1.data(), hop);
            for(int i=0; i < hop*F; ++i) tmp1[i] += res1[i];
            res1.swap(tmp1);
        }

        // c) final + skip + hf
        final_pw.run(res1.data(), finalH.data(), hop);
        skip_pw.run(in_hop.data(), skipH.data(), hop);
        hf_conv.run(in_hop.data(),    hfH.data(), hop);

        // d) merge + tanh vectorized
        vDSP_vadd(finalH.data(),1, skipH.data(),1, finalH.data(),1, hop);
        vDSP_vadd(finalH.data(),1, hfH  .data(),1, finalH.data(),1, hop);
        N = hop;
        vvtanhf(finalH.data(), finalH.data(), &N);

        // e) push to audio‐out ring
        sh->out.push(finalH.data(), hop);
    }
}

// ───────────────────────── PA CALLBACK ─────────────────────────────
static int paCallback(const void *inputBuffer,
                      void *outputBuffer,
                      unsigned long framesPerBuffer,
                      const PaStreamCallbackTimeInfo *,
                      PaStreamCallbackFlags,
                      void *userData)
{
  auto *sh = static_cast<Shared*>(userData);
  auto *in = static_cast<const float*>(inputBuffer);
  auto *out= static_cast<float*>(outputBuffer);
  size_t hop = framesPerBuffer;

  if(in) sh->in.push(in, hop);

  if(bypassFlag.load()){
    if(in){
      for(size_t i=0;i<hop;i++){
        float v = in[i];
        out[2*i]   = v;
        out[2*i+1] = v;
      }
    } else {
      memset(out,0,sizeof(float)*2*hop);
    }
    return paContinue;
  }

  float *temp = sh->tempOut.data();
  size_t got  = sh->out.pop(temp, hop);
  if(got < hop)
    memset(temp+got,0,(hop-got)*sizeof(float));
  for(size_t i=0;i<hop;i++){
    float v = temp[i];
    out[2*i]   = v;
    out[2*i+1] = v;
  }
  return paContinue;
}

// ───────────────────────── DEVICE LIST ────────────────────────────
static void listDevices(){
  Pa_Initialize();
  int n = Pa_GetDeviceCount();
  std::cout<<"PortAudio devices:\n";
  for(int i=0;i<n;i++){
    auto *d = Pa_GetDeviceInfo(i);
    std::cout<<" ["<<i<<"] "<<d->name
             <<" in="<<d->maxInputChannels
             <<" out="<<d->maxOutputChannels
             <<" sr="<<d->defaultSampleRate<<"\n";
  }
  Pa_Terminate();
  std::exit(0);
}

// ───────────────────────────── MAIN ──────────────────────────────
int main(int argc, char **argv){
    int  blockSize = MAX_HOP;
    int  inDev = -1, outDev = -1;
    bool listDev = false;
    std::string wdir = ".";

    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--list-devices"))      listDev = true;
        else if(!strcmp(argv[i],"--blocksize") && i+1<argc)
                                                  blockSize = atoi(argv[++i]);
        else if(!strcmp(argv[i],"--in-dev") && i+1<argc)
                                                  inDev = atoi(argv[++i]);
        else if(!strcmp(argv[i],"--out-dev") && i+1<argc)
                                                  outDev = atoi(argv[++i]);
        else if(!strcmp(argv[i],"--weights-dir") && i+1<argc)
                                                  wdir = argv[++i];
        else if(!strcmp(argv[i],"--int8"))         gUseInt8 = true;
        else {
            std::cerr<<"Unknown arg "<<argv[i]<<"\n";
            return 1;
        }
    }

    if(listDev) listDevices();
    if(inDev<0||outDev<0){
        std::cerr<<"Error: --in-dev and --out-dev required\n";
        listDevices();
    }
    if(blockSize > MAX_HOP){
        std::cerr<<"Error: --blocksize ≤ "<<MAX_HOP<<"\n";
        return 1;
    }

    std::cout<<"[MAIN] Weights dir: "<<wdir<<"\n";
    std::cout<<"[MAIN] Model mode: "<<(gUseInt8?"INT8":"FP32")<<"\n";
    if(gUseInt8) loadAllInt8(wdir);
    else          loadAllFp32(wdir);
    std::cout<<"[MAIN] Loaded weights.\n";

    PaError err = Pa_Initialize();
    if(err!=paNoError){
        std::cerr<<"Pa_Initialize failed: "<<Pa_GetErrorText(err)<<"\n";
        return 1;
    }

    auto *diIn  = Pa_GetDeviceInfo(inDev);
    auto *diOut = Pa_GetDeviceInfo(outDev);
    if(!diIn||!diOut){
        std::cerr<<"Bad device index\n";
        Pa_Terminate();
        return 1;
    }

    std::cout<<"[MAIN] In ["<<inDev<<"]  "<<diIn->name
             <<"  chIn="<<diIn->maxInputChannels
             <<"  sr="<<diIn->defaultSampleRate<<"\n";
    std::cout<<"[MAIN] Out["<<outDev<<"]  "<<diOut->name
             <<"  chOut="<<diOut->maxOutputChannels
             <<"  sr="<<diOut->defaultSampleRate<<"\n";

    int sampleRate = (int)std::lround(diIn->defaultSampleRate);
    std::cout<<"[MAIN] blockSize="<<blockSize
             <<"  RF="<<RECEPTIVE_FIELD<<"\n";

    // (2) allocate a big ring to absorb jitter
    size_t cap = 1<<14; // 16384 frames (~340 ms @48 kHz)
    Shared shared(cap);
    shared.tempOut.resize(blockSize);

    // (1) request low-latency I/O
    PaStreamParameters ip = {
        inDev, 1, paFloat32,
        diIn->defaultLowInputLatency, nullptr
    };
    PaStreamParameters op = {
        outDev, 2, paFloat32,
        diOut->defaultLowOutputLatency, nullptr
    };

    // non-blocking stdin for bypass toggle
    termios oldt,newt;
    tcgetattr(STDIN_FILENO,&oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON|ECHO);
    tcsetattr(STDIN_FILENO,TCSANOW,&newt);
    fcntl(STDIN_FILENO,F_SETFL,O_NONBLOCK);

    std::signal(SIGINT, signalHandler);

    // spawn the real-time inference thread
    std::thread infThread([&](){ inferenceThread(&shared, blockSize); });
    // spawn keyboard thread for bypass
    std::thread kbThread([&](){
      while(!gStopFlag){
        int c = getchar();
        if(c=='b'||c=='B'){
          bool now = !bypassFlag.load();
          bypassFlag.store(now);
          // reset audio-out read pointer
          shared.out.r.store(
            shared.out.w.load(std::memory_order_relaxed),
            std::memory_order_relaxed
          );
          std::cout<<"[MAIN] Bypass "<<(now?"ON":"OFF")<<"\n";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    });

    PaStream *stream = nullptr;
    err = Pa_OpenStream(&stream,
                       &ip, &op,
                       sampleRate, blockSize,
                       paNoFlag, paCallback, &shared);
    if(err!=paNoError){
      std::cerr<<"Pa_OpenStream failed: "<<Pa_GetErrorText(err)<<"\n";
      Pa_Terminate();
      return 1;
    }
    err = Pa_StartStream(stream);
    if(err!=paNoError){
      std::cerr<<"Pa_StartStream failed: "<<Pa_GetErrorText(err)<<"\n";
      Pa_CloseStream(stream);
      Pa_Terminate();
      return 1;
    }

    std::cout<<"[MAIN] Streaming… 'b'=bypass, Ctrl-C=quit\n";
    while(!gStopFlag) Pa_Sleep(100);

    std::cout<<"[MAIN] Stopping…\n";
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    infThread.join();
    kbThread.join();
    tcsetattr(STDIN_FILENO,TCSANOW,&oldt);
    std::cout<<"[MAIN] Done.\n";
    return 0;
}