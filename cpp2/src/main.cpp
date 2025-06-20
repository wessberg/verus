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
#include "cnpy.h"
#include <pthread/qos.h>
#include <Accelerate/Accelerate.h> // for cblas_sgemm
#include <portaudio.h>
#ifdef __APPLE__
  #include <sched.h>          // for SCHED_FIFO
  #include "pa_mac_core.h"    // PortAudio CoreAudio stream‐info
  #include <CoreAudio/CoreAudio.h>
  #include <CoreFoundation/CoreFoundation.h>
#endif













// ───────────────────────── PARAMETERS ─────────────────────────
static constexpr int FILTERS        = 44;
static constexpr int KERNEL         = 41;
static constexpr std::array<int,4> DILATIONS     = {1,2,4,8};
static constexpr int NUM_BLOCKS     = (int)DILATIONS.size();
static constexpr int MAX_HOP        = 15;
static constexpr int RECEPTIVE_FIELD=
    1 + (KERNEL-1)*(1+2+4+8);

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

  P1D() : inC(0), outC(0) {}
  P1D(int _inC, int _outC,
      std::vector<float>&& _W,
      std::vector<float>&& _b)
    : inC(_inC), outC(_outC),
      W(std::move(_W)),
      b(std::move(_b))
  {}

  //
  // in:  hop × inC   (row‐major, stride=1 between elements in a row)
  // W:   inC × outC (row‐major, stride=1)
  // out: hop × outC (row‐major, stride=1)
  //
  void run(const float *in, float *out, int hop) const {
    // 1) out = in (MxP) × W (PxN)
    //    where M = hop, P = inC, N = outC
    //
    //    vDSP_mmul(A, IA, B, IB, C, IC, M, N, P)
    //      IA, IB, IC = stride between elements in each ROW
    //
    vDSP_mmul(
      in,        /* A       */ 1,     /* IA = 1 since row‐major contiguous */
      W.data(),  /* B       */ 1,     /* IB = 1 */
      out,       /* C       */ 1,     /* IC = 1 */
      hop,       /* M = rows of A and C */
      outC,      /* N = cols of B and C */
      inC        /* P = cols of A, rows of B */
    );

    // 2) add bias b[oc] to each column oc in 'out'
    //
    for(int oc = 0; oc < outC; ++oc) {
      float bias = b[oc];
      // vDSP_vsadd(A, IA, &bias, C, IC, N)
      //   adds scalar (&bias) to each of N elements of A, stride IA, writes to C stride IC
      vDSP_vsadd(
        out + oc,   /* A       start at element (0,oc) */
        outC,       /* IA      to jump from one row to next you skip outC elems */
        &bias,      /* B       pointer to scalar to add */
        out + oc,   /* C       in‐place writeback */
        outC,       /* IC      same stride */
        hop         /* N       # of rows = hop */
      );
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
void inferenceThread(Shared *sh, int hop) {
    // 1) boost QoS on macOS
    #if defined(__APPLE__)
      // 1) bump to real‐time “audio” QoS if available
      #ifdef QOS_CLASS_AUDIO
        pthread_set_qos_class_self_np(QOS_CLASS_AUDIO, 0);
      #else
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
      #endif

      // 2) also try to grab an SCHED_FIFO real‐time thread priority
      struct sched_param sp;
      sp.sched_priority = sched_get_priority_max(SCHED_FIFO);
      if(pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0){
        std::cerr<<"[inferenceThread] WARNING: pthread_setschedparam(SCHED_FIFO) failed\n";
      }
    #endif

    const int RF  = RECEPTIVE_FIELD;
    const int SEG = RF + hop;
    const int F   = FILTERS;
    const int OC  = 2 * F;

    // ─── pre‐allocated buffers ────────────────────────────────────────
    std::vector<float> warm(SEG),
                       buf0(SEG*F), res(SEG*F),
                       sep(SEG*F), tmp(SEG*F),
                       pwsep(SEG*OC), gated(SEG*F),
                       finalW(SEG), skipW(SEG), hfW(SEG);

    std::vector<float> in_hop(hop),
                       buf1(hop*F), res1(hop*F),
                       sep1(hop*F), tmp1(hop*F),
                       gated1(hop*F),
                       finalH(hop), skipH(hop), hfH(hop);

    // ─── temps for gating vectorization ───────────────────────────────
    std::vector<float> tanhC(F), halfY(F), tanhY(F);
    const float half = 0.5f, one = 1.0f;

    // ─── warm‐up: fill “SEG” frames ───────────────────────────────────
    while(!gStopFlag && sh->in.available() < (size_t)SEG)
        std::this_thread::yield();
    if(gStopFlag) return;
    sh->in.pop(warm.data(), SEG);

    // a) initial 1×1
    init_pw.run(warm.data(), buf0.data(), SEG);
    std::copy(buf0.begin(), buf0.end(), res.begin());

    // b) gated dilated blocks (warm‐up)
    for(int b = 0; b < NUM_BLOCKS; ++b) {
        sep_conv[b].run(res.data(), sep.data(), SEG);
        pw_sep [b].run(sep.data(), pwsep.data(), SEG);

        // → vectorized gating:
        for(int t = 0; t < SEG; ++t) {
            const float *s    = pwsep.data() + t*OC;
            const float *cont = s;        // first F
            const float *gate = s + F;    // next  F
            float       *gOut = gated.data() + t*F;

            // 1) tanh(content) → tanhC
            vvtanhf(tanhC.data(), cont, &F);

            // 2) halfY = gate * 0.5
            vDSP_vsmul(gate, 1, &half, halfY.data(), 1, F);

            // 3) tanhY = tanh(halfY)
            vvtanhf(tanhY.data(), halfY.data(), &F);

            // 4) sigmoid = (tanhY + 1)*0.5  in‐place on tanhY
            vDSP_vsmsa(tanhY.data(), 1, &half, &half, tanhY.data(), 1, F);

            // 5) gated = tanhC * sigmoid
            vDSP_vmul(tanhC.data(), 1, tanhY.data(), 1, gOut,    1, F);
        }

        // 1×1 projection and residual combine
        pw2[b].run(gated.data(), tmp.data(), SEG);
        // tmp += res   →  vectorized
        vDSP_vadd(tmp.data(), 1, res.data(), 1, res.data(), 1, SEG*F);
    }

    // c) final 1×1 + skip + hf + merge + tanh
    final_pw.run(res.data(),  finalW.data(), SEG);
    skip_pw .run(warm.data(), skipW.data(), SEG);
    hf_conv .run(warm.data(),    hfW.data(), SEG);

    // finalW += skipW; finalW += hfW; finalW += warm
    vDSP_vadd(finalW.data(), 1, skipW.data(), 1, finalW.data(), 1, SEG);
    vDSP_vadd(finalW.data(), 1, hfW.  data(), 1, finalW.data(), 1, SEG);
    vDSP_vadd(finalW.data(), 1, warm. data(), 1, finalW.data(), 1, SEG);

    int Nseg = SEG;
    vvtanhf(finalW.data(), finalW.data(), &Nseg);

    // push the overlap back for steady‐state
    sh->in.push(warm.data() + RF, hop);

    // ─── steady‐state processing ──────────────────────────────────────
    while(!gStopFlag) {
        // wait for next hop
        while(!gStopFlag && sh->in.available() < (size_t)hop)
            std::this_thread::yield();
        if(gStopFlag) break;

        size_t got = sh->in.pop(in_hop.data(), hop);
        if(got < (size_t)hop)
            std::fill(in_hop.begin()+got, in_hop.end(), 0.f);

        // a) init 1×1
        init_pw.run(in_hop.data(), buf1.data(), hop);
        std::copy(buf1.begin(), buf1.end(), res1.begin());

        // b) gated dilated blocks (hop)
        for(int b = 0; b < NUM_BLOCKS; ++b) {
            sep_conv[b].run(res1.data(), sep1.data(), hop);
            pw_sep [b].run(sep1.data(), pwsep.data(), hop);

            // vectorized gating
            for(int t = 0; t < hop; ++t) {
                const float *s    = pwsep.data() + t*OC;
                const float *cont = s;
                const float *gate = s + F;
                float       *gOut= gated1.data() + t*F;

                vvtanhf(tanhC.data(), cont, &F);
                vDSP_vsmul(gate, 1, &half, halfY.data(), 1, F);
                vvtanhf(tanhY.data(), halfY.data(), &F);
                vDSP_vsmsa(tanhY.data(), 1, &half, &half, tanhY.data(), 1, F);
                vDSP_vmul(tanhC.data(), 1, tanhY.data(), 1, gOut,      1, F);
            }

            pw2[b].run(gated1.data(), tmp1.data(), hop);
            // tmp1 += res1   → vectorized
            vDSP_vadd(tmp1.data(), 1, res1.data(), 1, res1.data(), 1, hop*F);
        }

        // c) final + skip + hf + merge + tanh (hop)
        final_pw.run(res1.data(), finalH.data(), hop);
        skip_pw .run(in_hop.data(), skipH.data(), hop);
        hf_conv .run(in_hop.data(),   hfH.data(), hop);

        vDSP_vadd(finalH.data(), 1, skipH.data(), 1, finalH.data(), 1, hop);
        vDSP_vadd(finalH.data(), 1, hfH.data(),   1, finalH.data(), 1, hop);
        int Nhop = hop;
        vvtanhf(finalH.data(), finalH.data(), &Nhop);

        // d) push to audio‐out ring
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

struct DeviceInfo {
    int devId;
    std::string name;
    int maxInputChannels;
    int maxOutputChannels;
    double defaultSampleRate;
};

static std::vector<DeviceInfo> getDevices() {
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "Pa_Initialize failed: " << Pa_GetErrorText(err) << "\n";
        std::exit(1);
    }

    int n = Pa_GetDeviceCount();
    if (n < 0) {
        std::cerr << "Pa_GetDeviceCount failed: " << Pa_GetErrorText(n) << "\n";
        Pa_Terminate();
        std::exit(1);
    }

    std::vector<DeviceInfo> devices;
    devices.reserve(n);

    for (int i = 0; i < n; ++i) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
        if (!info) {
            std::cerr << "Pa_GetDeviceInfo failed for device " << i << "\n";
            continue;
        }
        DeviceInfo d;
        d.devId = i;
        d.name = info->name ? info->name : "";
        d.maxInputChannels  = info->maxInputChannels;
        d.maxOutputChannels = info->maxOutputChannels;
        d.defaultSampleRate = info->defaultSampleRate;
        devices.push_back(d);
    }

    Pa_Terminate();
    return devices;
}

static void listDevices(const std::vector<DeviceInfo> &devices) {
  int n = (int)devices.size();

  std::cout<<"PortAudio devices:\n";
  for(int i=0;i<n;i++){
    const auto &d = devices[i];
    std::cout<<"  "<<i<<": "<<d.name
             <<" (inputs: "<<d.maxInputChannels
             <<", outputs: "<<d.maxOutputChannels
             <<", sampleRate: "<<d.defaultSampleRate<<")\n";
  }

  std::exit(0);
}

// ───────────────────────────── MAIN ──────────────────────────────
int main(int argc, char **argv){
    int  blockSize = MAX_HOP;
    int  portAudioInputDeviceId = -1, portAudioOutputDeviceId = -1;
    DeviceInfo portAudioIputDevice, portAudioOutputDevice;

    bool listDev = false;
    std::string wdir = ".";

    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--list-devices"))      listDev = true;
        else if(!strcmp(argv[i],"--blocksize") && i+1<argc)
                                                  blockSize = atoi(argv[++i]);
        else if(!strcmp(argv[i],"--input-device") && i+1<argc)
                                                  portAudioInputDeviceId = atoi(argv[++i]);
        else if(!strcmp(argv[i],"--output-device") && i+1<argc)
                                                  portAudioOutputDeviceId = atoi(argv[++i]);
        else if(!strcmp(argv[i],"--weights-dir") && i+1<argc)
                                                  wdir = argv[++i];
        else if(!strcmp(argv[i],"--int8"))         gUseInt8 = true;
        else {
            std::cerr<<"Unknown arg "<<argv[i]<<"\n";
            return 1;
        }
    }

    auto devices = getDevices();

    if(listDev) {
      listDevices(devices);
    }

    if(portAudioInputDeviceId < 0 || portAudioOutputDeviceId < 0){
        std::cerr<<"Error: --input-device and --output-device required\n";
        listDevices(devices);
    }

    // Find and assign the matching devices
    for(const auto &d:devices){
        if(d.devId == portAudioInputDeviceId) {
            portAudioIputDevice = d;
        }
        if(d.devId == portAudioOutputDeviceId) {
            portAudioOutputDevice = d;
        }
    }

    if (portAudioIputDevice.devId < 0) {
        std::cerr<<"Error: input device ID "<<portAudioInputDeviceId
                 <<" not found\n";
        listDevices(devices);
    }

    if (portAudioOutputDevice.devId < 0) {
        std::cerr<<"Error: output device ID "<<portAudioOutputDeviceId
                 <<" not found\n";
        listDevices(devices);
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

    #ifdef __APPLE__
      //---------------------------------------------------------------------
      // On macOS we must poke the HAL buffer‐size property ourselves.
      // This forces the device to use exactly `blockSize` frames.
      //---------------------------------------------------------------------
      auto setMacHARDWareBufferSize = [&](bool isInput){
        AudioObjectID coreAudioDeviceID = 0;

        AudioObjectPropertyAddress rangeAddr = {
          kAudioDevicePropertyBufferFrameSizeRange,
          kAudioObjectPropertyScopeGlobal,
          kAudioObjectPropertyElementMain
        };

        // 1) Get the list of all AudioDeviceIDs
        AudioObjectPropertyAddress propAddr = {
            kAudioHardwarePropertyDevices,
            kAudioObjectPropertyScopeGlobal,
            kAudioObjectPropertyElementMain
        };
        UInt32 dataSize = 0;
        OSStatus deviceErr = AudioObjectGetPropertyDataSize(
            kAudioObjectSystemObject,
            &propAddr,
            0, nullptr,
            &dataSize
        );
        if (deviceErr) {
            std::cerr << "Error getting device data size: " << deviceErr << "\n";
        }

        int deviceCount = dataSize / sizeof(AudioDeviceID);
        std::vector<AudioDeviceID> devices(deviceCount);
        deviceErr = AudioObjectGetPropertyData(
            kAudioObjectSystemObject,
            &propAddr,
            0, nullptr,
            &dataSize,
            devices.data()
        );
        if (deviceErr) {
            std::cerr << "Error getting device list: " << deviceErr << "\n";
        }

        // Find the CoreAudio device that matches the input/output device
        for (auto devID : devices) {
            DeviceInfo portAudioDevice = isInput
                ? portAudioIputDevice
                : portAudioOutputDevice;
                
            AudioObjectPropertyAddress addr = {
                isInput
                    ? kAudioDevicePropertyStreams
                    : kAudioDevicePropertyStreams,
                kAudioObjectPropertyScopeGlobal,
                kAudioObjectPropertyElementMain
            };
            UInt32 size = sizeof(AudioObjectID);
            OSStatus err = AudioObjectGetPropertyData(
                devID, &addr, 0, nullptr, &size, &coreAudioDeviceID
            );
            if (err == noErr) {

              AudioObjectPropertyAddress nameAddr = {
                  kAudioObjectPropertyName,
                  kAudioObjectPropertyScopeGlobal,
                  kAudioObjectPropertyElementMain
              };
              CFStringRef cfName = nullptr;

              UInt32 nSize = sizeof(cfName);
              err = AudioObjectGetPropertyData(
                  devID,
                  &nameAddr,
                  0, nullptr,
                  &nSize,
                  &cfName
              );
              if (err == noErr && cfName) {
                  char nameBuf[256] = {0};
                  CFStringGetCString(cfName, nameBuf, sizeof(nameBuf), kCFStringEncodingUTF8);
                  CFRelease(cfName);
                  std::string nameStr(nameBuf);

                  if (nameStr == portAudioDevice.name) {
                      coreAudioDeviceID = devID;
                      break;
                  }
              }
            }
        }

        // Print the input/output device IDs
        std::cerr << "[MAIN] Matching CoreAudio "
                  << (isInput ? "input" : "output")
                  << " device ID: " << coreAudioDeviceID << "\n";

        UInt32 propSize = sizeof(AudioValueRange);
        AudioValueRange range = {0,0};

        if (AudioObjectGetPropertyData(
              coreAudioDeviceID,
              &rangeAddr,
              0, nullptr,
              &propSize,
              &range) == noErr)
        {
          std::cerr << "[MAIN] Device buffer-size range = "
                    << (unsigned)range.mMinimum
                    << " – "
                    << (unsigned)range.mMaximum
                    << " frames\n";
        }

         // 3) Set its buffer frame size (must be done BEFORE opening streams)
        UInt32 newBufferSize = blockSize;
        AudioObjectPropertyAddress bufAddr = {
            kAudioDevicePropertyBufferFrameSize,
            kAudioObjectPropertyScopeGlobal,
            kAudioObjectPropertyElementMain
        };
        deviceErr = AudioObjectSetPropertyData(
            coreAudioDeviceID,
            &bufAddr,
            0, nullptr,
            sizeof(newBufferSize),
            &newBufferSize
        );
        if (deviceErr) {
            std::cerr << "Error setting buffer size: " << deviceErr << "\n";
        }

        // 4) Verify
        UInt32 actualSize = 0;
        UInt32 asize = sizeof(actualSize);
        deviceErr = AudioObjectGetPropertyData(
            coreAudioDeviceID,
            &bufAddr,
            0, nullptr,
            &asize,
            &actualSize
        );
        if (deviceErr) {
            std::cerr << "Error reading back buffer size: " << deviceErr << "\n";
        }
        std::cout << "[MAIN] HAL buffer size is now: " << actualSize << " frames\n";
      };
      // set both input and output
      setMacHARDWareBufferSize(true);
      setMacHARDWareBufferSize(false);
    #endif

    PaError err = Pa_Initialize();
    if(err!=paNoError){
        std::cerr<<"Pa_Initialize failed: "<<Pa_GetErrorText(err)<<"\n";
        return 1;
    }

    auto *diIn  = Pa_GetDeviceInfo(portAudioInputDeviceId);
    auto *diOut = Pa_GetDeviceInfo(portAudioOutputDeviceId);

    std::cout<<"[MAIN] In ["<<portAudioInputDeviceId<<"]  "<<diIn->name
             <<"  chIn="<<diIn->maxInputChannels
             <<"  sr="<<diIn->defaultSampleRate<<"\n";
    std::cout<<"[MAIN] Out["<<portAudioOutputDeviceId<<"]  "<<diOut->name
             <<"  chOut="<<diOut->maxOutputChannels
             <<"  sr="<<diOut->defaultSampleRate<<"\n";

    int sampleRate = (int)std::lround(diIn->defaultSampleRate);

    double wantLatencySecs = double(blockSize) / sampleRate;

    std::cout<<"[MAIN] blockSize="<<blockSize
             <<"  RF="<<RECEPTIVE_FIELD<<"\n";

    // (2) allocate a big ring to absorb jitter
    size_t cap = 1<<14; // 16384 frames (~340 ms @48 kHz)
    Shared shared(cap);
    shared.tempOut.resize(blockSize);

    // (1) request low-latency I/O
    PaStreamParameters ip = {
        portAudioInputDeviceId, 1, paFloat32,
        wantLatencySecs, nullptr
    };
    PaStreamParameters op = {
        portAudioOutputDeviceId, 2, paFloat32,
        wantLatencySecs, nullptr
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
                   sampleRate,
                   (unsigned long)blockSize,   // ← directly request blockSize
                   paNoFlag,
                   paCallback,
                   &shared);
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

    auto *info = Pa_GetStreamInfo(stream);
    std::cerr<<"Actual I/O buffer = "
            << (int)std::round(info->inputLatency * sampleRate)
            <<" frames in, "
            << (int)std::round(info->outputLatency * sampleRate)
            <<" frames out\n";

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