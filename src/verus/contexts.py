from .config import (
    OFFLINE_TRAIN_LEFT_CTX,
    OFFLINE_TRAIN_RIGHT_CTX,
    REALTIME_COLOREQ_ORDER,
    REALTIME_SAMPLE_RATE,
    REALTIME_TRAIN_LEFT_CTX,
    _realtime_branch_left_contexts,
    _realtime_train_left_context,
    pickup_adapter_left_context,
)

def compute_realtime_streaming_context(model=None, sample_rate=REALTIME_SAMPLE_RATE, verbose=True):
    """
    Left context of the realtime model (samples @48k), right context 0 (purely causal). Derived from
    the architecture constants like REALTIME_TRAIN_LEFT_CTX: max over the branches (main stack, IR,
    pick attack incl. the analytic mask, pick resonance incl. the mask's envelopes) PLUS the colour
    EQ (it filters the branch sum) and the pickup adapter (it feeds every branch). The previous
    layer-introspection version only recognised training-graph layer classes and missed the pick
    mask entirely (it reported ~4.2k instead of ~8.5k samples). 'model' is accepted for API
    compatibility and ignored.
    """
    br = _realtime_branch_left_contexts()
    left_ctx = _realtime_train_left_context()
    if verbose:
        ms = 1000.0 * float(left_ctx) / float(sample_rate)
        print("[streaming] realtime LC (samples): "
              + ", ".join(f"{k}={v}" for k, v in br.items())
              + f", +coloreq={REALTIME_COLOREQ_ORDER - 1}, +adapter={pickup_adapter_left_context()}"
              + f" → {left_ctx} ({ms:.2f} ms)")
    return int(left_ctx), 0

def compute_streaming_ctx_constants(kind):
    """
    Streaming/windowed context purely from compile-time constants (no TF).
    Returns (left_ctx, right_ctx) in samples at the model's rate. Same derivation as the training
    segment layout (REALTIME_TRAIN_LEFT_CTX / OFFLINE_TRAIN_*_CTX): branch maximum incl. the pick
    mask, plus the colour EQ and the pickup adapter (both apply to every branch).
    """
    if kind == "realtime":
        return int(REALTIME_TRAIN_LEFT_CTX), 0
    elif kind == "offline":
        # Right context is the SUM of per-layer bounded look-aheads, not the per-layer bound.
        return int(OFFLINE_TRAIN_LEFT_CTX), int(OFFLINE_TRAIN_RIGHT_CTX)
    else:
        raise ValueError("kind must be 'offline' or 'realtime'")
