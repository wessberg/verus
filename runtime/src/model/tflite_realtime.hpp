#pragma once
// TFLite wrapper for the streaming (and windowed) realtime model.

#include "common.hpp"
#include "knobs.hpp"
#include "util.hpp"

// ────────────────────── NAME/DTYPE HELPERS ─────────────────────
static inline std::string normName(const char* n) {
  std::string s = n ? std::string(n) : std::string();
  if (s.rfind("serving_default_", 0) == 0)
    s = s.substr(std::strlen("serving_default_"));
  auto pos = s.find(':');
  if (pos != std::string::npos) s = s.substr(0, pos);
  // Remove TF scope prefixes like "StatefulPartitionedCall/..."
  auto slash = s.find_last_of('/');
  if (slash != std::string::npos) s = s.substr(slash+1);
  return s;
}
static inline bool isIntType(TfLiteType t) {
  return (t == kTfLiteInt32 || t == kTfLiteInt64 || t == kTfLiteUInt8 || t == kTfLiteInt16);
}

// ───────────────────────── TFLITE WRAPPER ─────────────────────────
class TFLiteRealtime {
public:
  // NOTE: the exported style output (out_style_) is now the EMA-smoothed probability itself
  // (the model EMAs the classifier logit and applies sigmoid after), so callers should use
  // the styleProb returned by invoke*() directly. The old getStyleEma() hack that read the
  // raw state slot has been removed — the state now stores the logit, not a probability.
  // Throws std::runtime_error on any load/validation failure (constructed on the main thread
  // before any stream or RT thread exists, so the caller can clean up and exit).
  explicit TFLiteRealtime(const std::string& modelPath) {
    model_ = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
    if (!model_) throw std::runtime_error("[TFLite] Cannot load model: " + modelPath);
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model_, resolver)(&interp_);
    if (!interp_) throw std::runtime_error("[TFLite] InterpreterBuilder failed");
    interp_->SetNumThreads(1);

    #ifdef TFLITE_WITH_XNNPACK
      {
        TfLiteXNNPackDelegateOptions opt = TfLiteXNNPackDelegateOptionsDefault();
        opt.num_threads = 1; // keep real-time friendly
        xnnpack_ = TfLiteXNNPackDelegateCreate(&opt);
        if (xnnpack_ && interp_->ModifyGraphWithDelegate(xnnpack_) != kTfLiteOk)
          std::cerr << "[TFLite] WARNING: XNNPACK delegate could not be applied; running the builtin kernels\n";
      }
    #endif

    if (interp_->AllocateTensors() != kTfLiteOk) throw std::runtime_error("[TFLite] AllocateTensors failed");

    // Build input/output maps
    for (int i : interp_->inputs()) {
      auto* t = interp_->tensor(i);
      in_map_[normName(t->name)] = i;
    }
    for (int i : interp_->outputs()) {
      auto* t = interp_->tensor(i);
      out_map_[normName(t->name)] = i;
    }

    // Required inputs
    in_piezo_   = findInput("realtime_piezo_input");
    in_st_true_ = findInput("realtime_style_true_input");
    in_st_force_= findInput("realtime_style_force_input");
    in_knob_pa_ = findInput("realtime_knob_pickattack");
    in_knob_pr_ = findInput("realtime_knob_pickreso");
    in_knob_ir_ = findInput("realtime_knob_ir");
    in_knob_dw_ = findInput("realtime_knob_drywet");

    // Optional inputs (resolved once; -1 when the export does not have them)
    in_pickup_        = findInput("realtime_pickup_input");
    in_knob_main_     = findInput("realtime_knob_main");
    in_knob_outgain_  = findInput("realtime_knob_outgain");
    in_knob_phaseinv_ = findInput("realtime_knob_phaseinv");

    // Legacy exports (before the master EQ moved to MasterEqIIR) still carry in-graph EQ knob
    // inputs: feed them their neutral, training-time values so the in-graph EQ is transparent.
    {
      const std::pair<const char*, float> legacy[] = {
        {"realtime_knob_lc_hz", 20.0f}, {"realtime_knob_lc_slope_dbpoct", 6.0f},
        {"realtime_knob_ls_gain_db", 0.0f}, {"realtime_knob_ls_hz", 200.0f}, {"realtime_knob_ls_slope_dbpoct", 12.0f},
        {"realtime_knob_mid_gain_db", 0.0f}, {"realtime_knob_mid_hz", 1000.0f}, {"realtime_knob_mid_q", 1.0f},
        {"realtime_knob_hs_gain_db", 0.0f}, {"realtime_knob_hs_hz", 4000.0f}, {"realtime_knob_hs_slope_dbpoct", 12.0f},
      };
      for (auto& kv : legacy) {
        int idx = findInput(kv.first);
        if (idx >= 0) in_legacy_eq_.emplace_back(idx, kv.second);
      }
      if (!in_legacy_eq_.empty())
        std::cout << "[TFLite] NOTE: legacy export with " << in_legacy_eq_.size()
                  << " in-graph master-EQ inputs (fed neutral values; the EQ runs post-model now)\n";
    }

    // Determine T
    auto* tin = (in_piezo_ >= 0) ? interp_->tensor(in_piezo_) : nullptr;
    if (!tin || tin->dims->size < 2) throw std::runtime_error("[TFLite] Unexpected/no piezo input; want [1,T,1]");
    if (tin->dims->size == 3 && (tin->dims->data[0] != 1 || tin->dims->data[2] != 1))
      throw std::runtime_error("[TFLite] Unexpected piezo shape; want [1,T,1]");
    T_ = tin->dims->data[1];

    // Discover streaming states by input names
    discoverStateInputs();

    // Pick outputs by shape/dtype
    classifyOutputsByShape();

    // Basic checks for required non-state I/O
    if (in_piezo_ < 0 || out_audio_ < 0 || in_st_true_ < 0 || in_st_force_ < 0 ||
        in_knob_pa_ < 0 || in_knob_pr_ < 0 || in_knob_ir_ < 0 || in_knob_dw_ < 0)
      throw std::runtime_error("[TFLite] Missing required I/O tensors (check model names)");
    for (int idx : {in_piezo_, in_st_true_, in_st_force_, in_knob_pa_, in_knob_pr_, in_knob_ir_, in_knob_dw_,
                    in_pickup_, in_knob_main_, in_knob_outgain_, in_knob_phaseinv_, out_audio_}) {
      if (idx >= 0 && interp_->tensor(idx)->type != kTfLiteFloat32)
        throw std::runtime_error(std::string("[TFLite] non-float32 tensor: ") + interp_->tensor(idx)->name);
    }
    for (const auto& s : stateSlots_) {
      if (interp_->tensor(s.input_idx)->type != kTfLiteFloat32)
        throw std::runtime_error("[TFLite] non-float32 state input: " + s.name);
    }

    if (ambiguousAudio_) {
      std::cerr << "[TFLite] WARNING: " << n3dOutputs_ << " 3-D float outputs and none named "
                   "'realtime_audio_out' — audio output selection is ambiguous; re-run --export-tflite.\n";
    }

    // The per-state lengths ('splits') are constants of the export: resolve the bundle → state
    // copy plan ONCE (the RT hop loop must not allocate) and validate it against the inputs.
    buildStateCopyPlan();

    std::cout<<"[TFLite] Ready. T="<<T_<<"  streaming="<<(isStreaming()?"true":"false")
             <<"  states="<<stateSlots_.size()<<"\n";
  }

  ~TFLiteRealtime() {
    // The delegate must outlive the interpreter that references it: destroy the interpreter first
    // (member destruction would otherwise run AFTER this body deleted the delegate).
    interp_.reset();
  #ifdef TFLITE_WITH_XNNPACK
      if (xnnpack_) TfLiteXNNPackDelegateDelete(xnnpack_);
  #endif
  }

  TFLiteRealtime(const TFLiteRealtime&) = delete;
  TFLiteRealtime& operator=(const TFLiteRealtime&) = delete;

  // Run a few zero hops so lazy one-time work (XNNPACK weight packing, first-touch of the arena)
  // happens before the stream starts instead of in the first RT hops; then clear the states.
  void warmUp(int hops = 8) {
    std::vector<float> zin(T_, 0.0f), zout(T_, 0.0f);
    float p = 0.0f;
    for (int i = 0; i < hops; ++i) {
      if (isStreaming()) invokeStreamingBlock(zin.data(), T_, zout.data(), p, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f);
      else               invokeWindowed(zin.data(), zout.data(), p, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f);
    }
    resetStates();
  }

  int T() const { return T_; }
  bool isStreaming() const { return !stateSlots_.empty(); }

  // Reset state buffers (zeros); call before starting stream
  void resetStates() {
    for (auto &s : stateSlots_) {
      std::fill(s.buf.begin(), s.buf.end(), 0.0f);
    }
  }

  void invokeStreamingBlock(const float* inBlockCore, int coreLen,
                          float* outBlockT, float &styleProb,
                          float pa_lin, float pr_lin, float ir_lin, float dry_frac,
                          float style_true, float style_force)
{
  // Prepare piezo [1,T,1]
  float* pz = interp_->typed_tensor<float>(in_piezo_);
  std::fill(pz, pz + T_, 0.0f);
  if (coreLen > 0) std::memcpy(pz, inBlockCore, sizeof(float) * std::min(coreLen, T_));

  // Scalars [1,1] for style (may be guided)
  *interp_->typed_tensor<float>(in_st_true_)  = style_true;
  *interp_->typed_tensor<float>(in_st_force_) = style_force;

  // Stems knobs (linear already)
  *interp_->typed_tensor<float>(in_knob_pa_) = pa_lin;
  *interp_->typed_tensor<float>(in_knob_pr_) = pr_lin;
  *interp_->typed_tensor<float>(in_knob_ir_) = ir_lin;
  *interp_->typed_tensor<float>(in_knob_dw_) = dry_frac;

  setOptionalScalarInputs();

  // Feed state inputs (if streaming)
  if (isStreaming()) {
    for (auto &s : stateSlots_) {
      float* dst = interp_->typed_tensor<float>(s.input_idx);
      std::memcpy(dst, s.buf.data(), sizeof(float) * s.len());
    }
  }

  if (interp_->Invoke() != kTfLiteOk) {
    invokeFailures_.fetch_add(1, std::memory_order_relaxed);   // reported by main(), never printed here
    std::memset(outBlockT, 0, sizeof(float) * T_);
    styleProb = 0.0f;
    return;
  }

  // Audio
  const float* y = interp_->typed_tensor<float>(out_audio_);
  std::memcpy(outBlockT, y, sizeof(float) * T_);

  // Style (2D [1,1] preferred)
  styleProb = 0.0f;
  if (out_style_ >= 0) {
    auto* st = interp_->tensor(out_style_);
    if (st && st->type == kTfLiteFloat32) {
      if (st->dims->size == 2 && st->dims->data[0] == 1 && st->dims->data[1] == 1) {
        styleProb = *interp_->typed_tensor<float>(out_style_);
      } else {
        styleProb = interp_->typed_tensor<float>(out_style_)[0];
      }
    }
  }

  // Update states from the packed bundle (plan precomputed in buildStateCopyPlan: no allocation,
  // no splits decoding per hop)
  if (isStreaming() && out_bundle_ >= 0) {
    const float* bundle = interp_->typed_tensor<float>(out_bundle_);
    for (const auto& c : copyPlan_) {
      std::memcpy(stateSlots_[c.slot].buf.data(), bundle + c.offset, sizeof(float) * c.count);
    }
  }
}

  void invokeWindowed(const float* inWindowT, float* outWindowT, float &styleProb,
                    float pa_lin, float pr_lin, float ir_lin, float dry_frac,
                    float style_true, float style_force)
{
  float* pz = interp_->typed_tensor<float>(in_piezo_);
  std::memcpy(pz, inWindowT, sizeof(float) * T_);

  *interp_->typed_tensor<float>(in_st_true_)  = style_true;
  *interp_->typed_tensor<float>(in_st_force_) = style_force;

  *interp_->typed_tensor<float>(in_knob_pa_)  = pa_lin;
  *interp_->typed_tensor<float>(in_knob_pr_)  = pr_lin;
  *interp_->typed_tensor<float>(in_knob_ir_)  = ir_lin;
  *interp_->typed_tensor<float>(in_knob_dw_)  = dry_frac;

  setOptionalScalarInputs();

  if (interp_->Invoke() != kTfLiteOk) {
    invokeFailures_.fetch_add(1, std::memory_order_relaxed);
    std::memset(outWindowT, 0, sizeof(float) * T_);
    styleProb = 0.0f;
    return;
  }
  const float* y = interp_->typed_tensor<float>(out_audio_);
  std::memcpy(outWindowT, y, sizeof(float) * T_);

  styleProb = 0.0f;
  if (out_style_ >= 0) {
    auto* st = interp_->tensor(out_style_);
    if (st && st->type == kTfLiteFloat32) {
      if (st->dims->size == 2 && st->dims->data[0] == 1 && st->dims->data[1] == 1) {
        styleProb = *interp_->typed_tensor<float>(out_style_);
      } else {
        styleProb = interp_->typed_tensor<float>(out_style_)[0];
      }
    }
  }
}

  uint32_t invokeFailures() const { return invokeFailures_.load(std::memory_order_relaxed); }

private:
  struct StateSlot {
    std::string name;
    int input_idx = -1;
    int L = 0;
    int C = 1;
    std::vector<float> buf;
    int len() const { return L * C; }
  };
  struct StateCopy { size_t slot; size_t offset; size_t count; };

  // Bundle layout = states in lexicographic input-name order (stateSlots_ is sorted the same way),
  // each 'splits[i]' floats long. The splits output is a constant tensor; read it after one
  // invoke (constant outputs are materialised by then) and fall back to the input shapes.
  void buildStateCopyPlan() {
    copyPlan_.clear();
    if (!isStreaming()) return;
    if (out_bundle_ < 0) throw std::runtime_error("[TFLite] streaming model without a states bundle output");
    // Invoke once with zero inputs so constant outputs (splits) are populated.
    for (int i : interp_->inputs()) {
      TfLiteTensor* t = interp_->tensor(i);
      if (t->type == kTfLiteFloat32 && t->data.raw) std::memset(t->data.raw, 0, t->bytes);
    }
    if (interp_->Invoke() != kTfLiteOk) throw std::runtime_error("[TFLite] warm-up Invoke failed");

    std::vector<int64_t> lens;
    if (out_splits_ >= 0) {
      const TfLiteTensor* sp = interp_->tensor(out_splits_);
      int N = 1; for (int i = 0; i < sp->dims->size; ++i) N *= sp->dims->data[i];
      for (int i = 0; i < N; ++i) {
        if (sp->type == kTfLiteInt32)      lens.push_back(interp_->typed_tensor<int32_t>(out_splits_)[i]);
        else if (sp->type == kTfLiteInt64) lens.push_back(interp_->typed_tensor<int64_t>(out_splits_)[i]);
      }
    }
    const TfLiteTensor* bt = interp_->tensor(out_bundle_);
    size_t total = 1;
    for (int i = 0; i < bt->dims->size; ++i) total *= size_t(bt->dims->data[i]);

    size_t offset = 0;
    for (size_t i = 0; i < stateSlots_.size(); ++i) {
      const size_t want = size_t(stateSlots_[i].len());
      const size_t seg  = (i < lens.size()) ? size_t(std::max<int64_t>(0, lens[i])) : want;
      if (seg != want)
        std::cerr << "[TFLite] WARNING: state '" << stateSlots_[i].name << "' bundle length " << seg
                  << " != input length " << want << "\n";
      const size_t n = std::min(want, seg);
      if (offset + n > total) throw std::runtime_error("[TFLite] states bundle shorter than the state inputs");
      if (n > 0) copyPlan_.push_back({i, offset, n});
      offset += seg;
    }
    if (offset != total)
      std::cerr << "[TFLite] WARNING: bundle length " << total << " != sum of state lengths " << offset << "\n";
  }

  int findInput(const std::string& name) const {
    auto it = in_map_.find(name);
    return (it != in_map_.end()) ? it->second : -1;
  }

  // Optional scalar inputs; indices were resolved once in the constructor (no per-hop lookups).
  inline void setOptionalScalarInputs() {
    if (in_pickup_ >= 0)        *interp_->typed_tensor<float>(in_pickup_)        = gKnobs.pickup01.load();
    if (in_knob_main_ >= 0)     *interp_->typed_tensor<float>(in_knob_main_)     = db2lin(gKnobs.main_dB.load());
    if (in_knob_outgain_ >= 0)  *interp_->typed_tensor<float>(in_knob_outgain_)  = db2lin(gKnobs.outGain_dB.load());
    if (in_knob_phaseinv_ >= 0) *interp_->typed_tensor<float>(in_knob_phaseinv_) = float(gKnobs.phaseInvert01.load());
    for (auto &kv : in_legacy_eq_) *interp_->typed_tensor<float>(kv.first) = kv.second;
  }

  void discoverStateInputs() {
    // Collect realtime_state_* inputs and sort them lexicographically by name
    std::vector<StateSlot> tmp;
    for (auto &kv : in_map_) {
      const std::string& nm = kv.first;
      if (nm.rfind("realtime_state_", 0) == 0) {
        int idx = kv.second;
        const TfLiteTensor* t = interp_->tensor(idx);
        if (!t || t->dims->size < 2) continue;
        StateSlot s;
        s.name = nm;
        s.input_idx = idx;
        // Expect shape [1, L, C] or [1, L]
        if (t->dims->size == 3) {
          s.L = t->dims->data[1];
          s.C = t->dims->data[2];
        } else if (t->dims->size == 2) {
          s.L = t->dims->data[1];
          s.C = 1;
        } else {
          continue;
        }
        s.buf.resize(std::max(1, s.L*s.C), 0.0f);
        tmp.push_back(std::move(s));
      }
    }
    std::sort(tmp.begin(), tmp.end(), [](const StateSlot& a, const StateSlot& b){
      return a.name < b.name;
    });
    stateSlots_.swap(tmp);
  }

  void classifyOutputsByShape() {
  out_audio_  = -1;
  out_style_  = -1;
  out_bundle_ = -1;
  out_splits_ = -1;

  size_t bestAudioTotal  = 0;
  size_t bestBundleTotal = 0;

  // 1) Audio = the 3D float output. Prefer the exported name when TFLite kept it; otherwise the
  //    largest 3D tensor. Current exports have exactly one 3D output (the old "dry" copy of the
  //    input was removed because it made this choice ambiguous); warn if that is not the case.
  n3dOutputs_ = 0;
  for (int idx : interp_->outputs()) {
    TfLiteTensor* t = interp_->tensor(idx);
    if (!t || !t->dims) continue;
    if (t->type == kTfLiteFloat32 && t->dims->size == 3) {
      ++n3dOutputs_;
      size_t total = 1;
      for (int i = 0; i < t->dims->size; ++i) total *= t->dims->data[i];
      if (total > bestAudioTotal) {
        bestAudioTotal = total;
        out_audio_ = idx;
      }
    }
  }
  {
    auto it = out_map_.find("realtime_audio_out");
    if (it != out_map_.end()) { out_audio_ = it->second; ambiguousAudio_ = false; }
    else ambiguousAudio_ = (n3dOutputs_ > 1);
  }

  // 2) Style = 2D float (1,1) if present
  for (int idx : interp_->outputs()) {
    TfLiteTensor* t = interp_->tensor(idx);
    if (!t || !t->dims) continue;
    if (t->type == kTfLiteFloat32 && t->dims->size == 2 &&
        t->dims->data[0] == 1 && t->dims->data[1] == 1) {
      out_style_ = idx;
      break;
    }
  }

  // 3) Bundle = largest 2D float (excluding (1,1))
  for (int idx : interp_->outputs()) {
    TfLiteTensor* t = interp_->tensor(idx);
    if (!t || !t->dims) continue;
    if (t->type == kTfLiteFloat32 && t->dims->size == 2) {
      int r = t->dims->data[0], c = t->dims->data[1];
      if (!(r == 1 && c == 1)) {
        size_t total = static_cast<size_t>(r) * static_cast<size_t>(c);
        if (total > bestBundleTotal) {
          bestBundleTotal = total;
          out_bundle_ = idx;
        }
      }
    }
  }

  // 4) Splits = int vector (1,Nstates) or (Nstates,)
  const int nStates = static_cast<int>(stateSlots_.size());
  for (int idx : interp_->outputs()) {
    TfLiteTensor* t = interp_->tensor(idx);
    if (!t || !t->dims) continue;
    if (!isIntType(t->type)) continue;
    int dims = t->dims->size;
    if (dims == 2 && t->dims->data[0] == 1 && t->dims->data[1] == nStates) {
      out_splits_ = idx;
      break;
    }
    if (dims == 1 && t->dims->data[0] == nStates) {
      out_splits_ = idx;
      break;
    }
  }

  // Fallbacks if something still missing:
  if (out_audio_ < 0) {
    // pick any 3D float
    for (int idx : interp_->outputs()) {
      TfLiteTensor* t = interp_->tensor(idx);
      if (t && t->type == kTfLiteFloat32 && t->dims && t->dims->size == 3) {
        out_audio_ = idx;
        break;
      }
    }
    if (out_audio_ < 0) {
      // as a last resort, choose the largest tensor
      size_t best = 0;
      for (int idx : interp_->outputs()) {
        TfLiteTensor* t = interp_->tensor(idx);
        if (!t || !t->dims) continue;
        size_t total = 1;
        for (int i = 0; i < t->dims->size; ++i) total *= t->dims->data[i];
        if (total > best) { best = total; out_audio_ = idx; }
      }
    }
  }
  if (out_style_ < 0) {
    // choose any small 2D/1D float tensor
    for (int idx : interp_->outputs()) {
      TfLiteTensor* t = interp_->tensor(idx);
      if (t && t->type == kTfLiteFloat32 && t->dims && t->dims->size <= 2) {
        out_style_ = idx;
        break;
      }
    }
  }
}

  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter>     interp_;
  #ifdef TFLITE_WITH_XNNPACK
    TfLiteDelegate* xnnpack_ = nullptr;
  #else
    void* xnnpack_ = nullptr;
  #endif

  std::unordered_map<std::string, int> in_map_, out_map_;
  int in_piezo_=-1, in_st_true_=-1, in_st_force_=-1, in_knob_pa_=-1, in_knob_pr_=-1, in_knob_ir_=-1, in_knob_dw_=-1;
  int in_pickup_=-1, in_knob_main_=-1, in_knob_outgain_=-1, in_knob_phaseinv_=-1;
  std::vector<std::pair<int,float>> in_legacy_eq_;
  int out_audio_=-1, out_style_=-1, out_bundle_=-1, out_splits_=-1;
  int n3dOutputs_ = 0;
  bool ambiguousAudio_ = false;
  int T_ = 0;

  std::vector<StateSlot> stateSlots_;
  std::vector<StateCopy> copyPlan_;
  std::atomic<uint32_t> invokeFailures_{0};
};
