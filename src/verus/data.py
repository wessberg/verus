import glob
import math
import numpy as np
import os
import random
import re
import scipy.signal as spsig
import shutil
import soundfile as sf
import sys
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable  # pyright: ignore[reportMissingImports]

from .config import (
    SEED,
    CHUNK_SEC,
    DEBUG,
    FINETUNE_TARGET_DENOISE_BETA,
    FINETUNE_TARGET_DENOISE_ENABLE,
    FINETUNE_TARGET_DENOISE_FLOOR_DB,
    FINETUNE_TARGET_DENOISE_HOP_MS,
    FINETUNE_TARGET_DENOISE_MIN_FRAMES,
    FINETUNE_TARGET_DENOISE_NOISE_Q,
    FINGER_CHUNK_RATIO,
    FINGER_SRC_DIR,
    FINGER_TGT_DIR,
    INPUT_DIR,
    KD_TEACHER_LEFT_CTX_SAMPLES,
    KD_TEACHER_RIGHT_CTX_SAMPLES,
    OFFLINE_SAMPLE_RATE,
    OFFLINE_SEG_LEN,
    PICKUP_DIR_NAMES,
    PICKUP_MAGNETIC,
    PICKUP_PIEZO,
    PICK_CHUNK_RATIO,
    PICK_SRC_DIR,
    PICK_TGT_DIR,
    REALTIME_SAMPLE_RATE,
    REALTIME_SEG_LEN,
    SEGMENT_GENERATOR_MIN_AMP_THRESH,
    STYLE_DIR_NAMES,
)
from .dsp import (
    resample_to_model_sr,
)
from .knobs import (
    prepare_training_knob_feeds,
)

def _extract_epoch_from_ckpt(path):
    fn = os.path.basename(path)
    m = re.search(r'epoch_(\d+)', fn)
    return int(m.group(1)) if m else 0

# ────────────────────────────── DATA LOADING ───────────────────────────────────
# The corpus is never held in RAM: load_training_data() only scans the pairs, and training
# segments are read from the WAV files on demand (_LazyMono). As float32 the 4 × 1 h × 96 kHz
# takes are 5.5 GB, which on the 16 GB dev Mac pushed training into swap until the disk filled.
pairs_by_group = {}   # (pickup_name, style_name) -> [(src, tgt), ...]
picked_pairs = []     # every pick-style pair, across pickups (legacy consumers)
finger_pairs = []     # every finger-style pair, across pickups
all_pairs = []
_file_sr_cache = {}
_mono_view_cache = {}
_training_data_loaded = False

def _scan_pair_dir(src_dir, tgt_dir):
    pairs = []
    for src in sorted(glob.glob(os.path.join(src_dir, "*_source.wav"))):
        tgt = os.path.join(tgt_dir, os.path.basename(src).replace("_source.wav", "_target.wav"))
        if os.path.exists(tgt):
            pairs.append((src, tgt))
        else:
            print("Warning: no target for", src)
    return pairs

def load_training_data():
    """
    Lazily scan ./input for training pairs (headers only; audio is read per segment, see _LazyMono).

    Canonical layout: input/<pickup>/<style>/{source,target} with pickup in {piezo, magnetic}
    and style in {pick, finger}. The legacy layout input/<style>/{source,target} is treated
    as piezo.

    TODO(data): once simultaneous piezo+magnetic+mic triplet recordings exist, pair the two
    sources of the same take here and add a cross-pickup consistency loss during training
    (model(piezo_src) ≈ model(magnetic_src) for the same performance).
    """
    global _training_data_loaded
    if _training_data_loaded:
        return
    _training_data_loaded = True

    for pickup in PICKUP_DIR_NAMES:
        for style in STYLE_DIR_NAMES:
            src_dir = os.path.join(INPUT_DIR, pickup, style, "source")
            tgt_dir = os.path.join(INPUT_DIR, pickup, style, "target")
            pairs_by_group[(pickup, style)] = _scan_pair_dir(src_dir, tgt_dir)

    # Legacy piezo layout
    for style, (sdir, tdir) in (("pick", (PICK_SRC_DIR, PICK_TGT_DIR)),
                                ("finger", (FINGER_SRC_DIR, FINGER_TGT_DIR))):
        for pair in _scan_pair_dir(sdir, tdir):
            if pair not in pairs_by_group[("piezo", style)]:
                pairs_by_group[("piezo", style)].append(pair)

    picked_pairs[:] = [p for pu in PICKUP_DIR_NAMES for p in pairs_by_group[(pu, "pick")]]
    finger_pairs[:] = [p for pu in PICKUP_DIR_NAMES for p in pairs_by_group[(pu, "finger")]]
    all_pairs[:] = picked_pairs + finger_pairs

    for src, tgt in all_pairs:
        info_src = sf.info(src)
        info_tgt = sf.info(tgt)
        if info_src.samplerate != info_tgt.samplerate:
            print(f"Warning: sample-rate mismatch for pair {src} vs {tgt}: {info_src.samplerate} vs {info_tgt.samplerate}")
        _file_sr_cache[src] = info_src.samplerate
        _file_sr_cache[tgt] = info_tgt.samplerate

def _file_sr(path):
    sr = _file_sr_cache.get(path)
    if sr is None:
        sr = sf.info(path).samplerate
        _file_sr_cache[path] = sr
    return int(sr)

class _LazyMono:
    """
    Channel 0 of a WAV file at model_sr, read from disk per slice (supports len() and [a:b]).
    Slices equal the whole-file array the old in-RAM cache held: exact at the file's rate; otherwise
    resample_poly over a window wider than its filter, started on a polyphase-aligned sample, so
    the interior matches the whole-file resample and there are no per-segment edge transients.
    Beyond the file edges it zero-pads like resample_poly does.
    """
    def __init__(self, path, model_sr):
        self.path = path
        self.frames = int(sf.info(path).frames)
        file_sr = _file_sr(path)
        g = math.gcd(file_sr, int(model_sr))
        self.up, self.down = int(model_sr) // g, file_sr // g
        self.n = -(-self.frames * self.up // self.down)          # == len(resample_poly(x, up, down))
        # resample_poly's filter half-length is 10*max(up, down) at the upsampled rate; margin in
        # whole blocks of `down` file samples (↔ `up` output samples) keeps the polyphase alignment
        self.k = -(-(10 * max(self.up, self.down) // self.up + 2) // self.down)

    def __len__(self):
        return self.n

    def _read(self, a, b):
        """File samples [a, b), zero outside the file."""
        out = np.zeros((b - a,), np.float32)
        lo, hi = max(a, 0), min(b, self.frames)
        if hi > lo:
            x, _ = sf.read(self.path, start=lo, frames=hi - lo, dtype='float32', always_2d=True)
            out[lo - a: lo - a + x.shape[0]] = x[:, 0]
        return out

    def __getitem__(self, sl):
        if not isinstance(sl, slice):
            raise TypeError("_LazyMono supports contiguous slices only")
        start, stop, step = sl.indices(self.n)
        if step != 1:
            raise TypeError("_LazyMono supports contiguous slices only")
        if stop <= start:
            return np.zeros((0,), np.float32)
        if self.up == self.down:
            return self._read(start, stop)
        b0 = start // self.up - self.k                           # first block (output b0*up ↔ file b0*down)
        b1 = -(-stop // self.up) + self.k
        y = resample_to_model_sr(self._read(b0 * self.down, b1 * self.down), self.down, self.up)
        o = start - b0 * self.up
        return y[o: o + (stop - start)]

def _mono_at_sr(path, model_sr):
    """Whole-file mono view of a training file at model_sr (read per slice from disk, see _LazyMono)."""
    key = (path, int(model_sr))
    view = _mono_view_cache.get(key)
    if view is None:
        view = _mono_view_cache[key] = _LazyMono(path, model_sr)
    return view

def _path_components(path):
    return [p.lower() for p in os.path.normpath(str(path)).split(os.sep)]

def pickup_flag_for_path(path):
    # Directory-component match (a substring match would also hit 'magnetic' inside a file name)
    return PICKUP_MAGNETIC if "magnetic" in _path_components(path) else PICKUP_PIEZO

def style_flag_for_path(path):
    """1.0 for pick takes, 0.0 for finger takes, None if the path names neither style directory."""
    comps = _path_components(path)
    if "pick" in comps:
        return 1.0
    if "finger" in comps:
        return 0.0
    return None

def build_chunks(pairs_list, seg_len, sample_rate):
    """
    Create chunks using the source file sample-rate (not the model's sample-rate).
    This prevents incorrect chunking when training the 48 kHz realtime model on 96 kHz files.
    """
    chunks = []
    for src, tgt in pairs_list:
        info = sf.info(src)
        file_sr = info.samplerate
        total = info.frames
        chunk_len = int(CHUNK_SEC * file_sr)

        n_full = total // chunk_len
        for i in range(n_full):
            chunks.append((src, tgt, i * chunk_len, chunk_len))

        rem = total - n_full * chunk_len
        # Ensure the remainder is long enough to produce a model segment after resampling
        # Minimum source samples needed to produce seg_len model samples:
        need_src_len = int(math.ceil(seg_len * file_sr / float(sample_rate)))
        if rem >= need_src_len:
            chunks.append((src, tgt, n_full * chunk_len, rem))
    return chunks
        
def assert_has_pairs ():
    load_training_data()
    if not all_pairs:
        print("ERROR: No training pairs found. Expected files named *_source.wav and *_target.wav "
            "under input/<pickup>/<style>/{source,target} (pickup: piezo|magnetic, style: pick|finger) "
            "or the legacy input/<style>/{source,target} layout.")
        sys.exit(1)

def _read_pair_segment(src_path, tgt_path, sample_rate, offset_sec, sec):
    """Read [offset, offset+sec) of a source/target pair as mono arrays at sample_rate."""
    info = sf.info(src_path)
    file_sr = info.samplerate
    start = int(round(offset_sec * file_sr))
    frames = int(round(sec * file_sr))
    x2d, _ = sf.read(src_path, start=start, frames=frames, dtype='float32', always_2d=True)
    x = x2d[:, 0]
    x_mod = x if file_sr == sample_rate else resample_to_model_sr(x, file_sr, sample_rate)
    y_mod = None
    if tgt_path is not None and os.path.exists(tgt_path):
        y2d, _ = sf.read(tgt_path, start=start, frames=frames, dtype='float32', always_2d=True)
        y = y2d[:, 0]
        y_mod = y if file_sr == sample_rate else resample_to_model_sr(y, file_sr, sample_rate)
    return x_mod, y_mod

def precache_finetune_resampled_pair(x_full, y_full, file_sr, sr_rt=REALTIME_SAMPLE_RATE, sr_off=OFFLINE_SAMPLE_RATE):
    """
    Pre-resample the normalized pair once for finetune to avoid per-batch resample_poly.
    x_full,y_full: float32 arrays (N,C).
    Returns dict with mono arrays at 96k and 48k.
    """
    x_m = x_full[:, 0].astype(np.float32, copy=False)
    y_m = y_full[:, 0].astype(np.float32, copy=False)

    def to_sr(sig, src_sr, dst_sr):
        return sig if int(src_sr) == int(dst_sr) else resample_to_model_sr(sig, int(src_sr), int(dst_sr))

    x96 = to_sr(x_m, file_sr, sr_off).astype(np.float32, copy=False)
    y96 = to_sr(y_m, file_sr, sr_off).astype(np.float32, copy=False)
    x48 = to_sr(x_m, file_sr, sr_rt).astype(np.float32, copy=False)
    y48 = to_sr(y_m, file_sr, sr_rt).astype(np.float32, copy=False)
    return {"x96": x96, "y96": y96, "x48": x48, "y48": y48}

def _slice_zero_padded(arr, start, end):
    """arr[start:end] with zeros wherever the range falls outside the array."""
    n = int(end) - int(start)
    out = np.zeros((n,), dtype=np.float32)
    a = max(0, int(start)); b = min(len(arr), int(end))
    if b > a:
        out[a - int(start): b - int(start)] = arr[a:b]
    return out

def kd_teacher_segment_len(seg_len_rt, factor=2):
    """Length of the teacher (96k) copy of a realtime training segment incl. teacher context."""
    return int(KD_TEACHER_LEFT_CTX_SAMPLES + int(factor) * int(seg_len_rt) + KD_TEACHER_RIGHT_CTX_SAMPLES)

def segment_generator_from_arrays_dual(seg_len_rt, x48, y48, x96, is_pick_chunks,
                                       factor=2, min_amp=SEGMENT_GENERATOR_MIN_AMP_THRESH):
    """
    Simple dual-rate generator backed by already-resampled arrays (48k and 96k).
    - seg_len_rt: segment length at realtime rate (48k).
    - x48, y48: mono float32 arrays at 48k.
    - x96: mono float32 array at 96k (teacher input, extended by the KD teacher context so the
      non-causal teacher is fully contexted over the student's segment).
    - is_pick_chunks: bool; unused for finetune labels (style disabled), but kept for symmetry.
    """
    T48 = len(x48)
    L96 = seg_len_rt * int(factor)
    while True:
        if T48 <= 1:
            raise RuntimeError("Input array too short for segment_generator_from_arrays_dual")
        s48 = random.randrange(max(T48 - seg_len_rt, 1))
        e48 = s48 + seg_len_rt

        xs48 = x48[s48:e48]
        ys48 = y48[s48:e48]

        if np.max(np.abs(xs48)) < float(min_amp):
            continue

        s96 = s48 * int(factor)
        xs96 = _slice_zero_padded(x96, s96 - KD_TEACHER_LEFT_CTX_SAMPLES,
                                  s96 + L96 + KD_TEACHER_RIGHT_CTX_SAMPLES)

        # Pad to exact lengths if needed
        if len(xs48) < seg_len_rt:
            p = seg_len_rt - len(xs48)
            xs48 = np.pad(xs48, (0, p))
            ys48 = np.pad(ys48, (0, p))

        style_flag = 1.0 if is_pick_chunks else 0.0  # ignored downstream (style loss disabled)
        yield (xs48[:, None].astype(np.float32),
               xs96[:, None].astype(np.float32)), \
              (ys48[:, None].astype(np.float32),
               np.float32(style_flag))

def segment_generator_from_arrays(seg_len, x, y, is_pick_chunks, min_amp=SEGMENT_GENERATOR_MIN_AMP_THRESH):
    """
    Simple, fast generator backed by already-resampled arrays.
    """
    T = len(x)
    while True:
        if T <= 1:
            raise RuntimeError("Input array too short for segment_generator_from_arrays")
        s = random.randrange(max(T - seg_len, 1))
        xs = x[s:s + seg_len]
        ys = y[s:s + seg_len]
        if np.max(np.abs(xs)) < float(min_amp):
            continue
        if len(xs) < seg_len:
            pad = seg_len - len(xs)
            xs = np.pad(xs, (0, pad))
            ys = np.pad(ys, (0, pad))
        style_flag = 1.0 if is_pick_chunks else 0.0
        yield xs[:, None].astype(np.float32), (ys[:, None].astype(np.float32), np.float32(style_flag))

def make_mixed_ds_from_arrays_realtime_dual(x48, x96, y48, batch_size, seg_len_rt=REALTIME_SEG_LEN):
    """
    Dual-rate dataset for finetuning realtime model with a 96k teacher.
    Returns a dataset of ((x48, x96), (y48, style_flag)).
    """
    factor = OFFLINE_SAMPLE_RATE // REALTIME_SAMPLE_RATE
    assert factor >= 2 and OFFLINE_SAMPLE_RATE == 96000 and REALTIME_SAMPLE_RATE == 48000, \
        "Expected 96k offline and 48k realtime rates with integer ratio."

    sig_x = (
        tf.TensorSpec((seg_len_rt, 1), tf.float32),                            # realtime 48k
        tf.TensorSpec((kd_teacher_segment_len(seg_len_rt, factor), 1), tf.float32)  # offline 96k (+ctx)
    )
    sig_y = (tf.TensorSpec((seg_len_rt, 1), tf.float32), tf.TensorSpec((), tf.float32))

    ds_pu = tf.data.Dataset.from_generator(
        lambda: segment_generator_from_arrays_dual(seg_len_rt, x48, y48, x96, is_pick_chunks=True, factor=factor),
        output_signature=(sig_x, sig_y)
    )
    ds_fu = tf.data.Dataset.from_generator(
        lambda: segment_generator_from_arrays_dual(seg_len_rt, x48, y48, x96, is_pick_chunks=False, factor=factor),
        output_signature=(sig_x, sig_y)
    )
    return tf.data.Dataset.sample_from_datasets([ds_pu, ds_fu], weights=[PICK_CHUNK_RATIO, FINGER_CHUNK_RATIO], seed=SEED)\
             .batch(batch_size).prefetch(tf.data.AUTOTUNE)

def make_mixed_ds_from_arrays_offline(x96, y96, batch_size, seg_len=OFFLINE_SEG_LEN):
    sig_x = tf.TensorSpec((seg_len, 1), tf.float32)
    sig_y_audio = tf.TensorSpec((seg_len, 1), tf.float32)
    sig_y_style = tf.TensorSpec((), tf.float32)
    sig_y = (sig_y_audio, sig_y_style)

    ds_pu = tf.data.Dataset.from_generator(
        lambda: segment_generator_from_arrays(seg_len, x96, y96, is_pick_chunks=True),
        output_signature=(sig_x, sig_y)
    )
    ds_fu = tf.data.Dataset.from_generator(
        lambda: segment_generator_from_arrays(seg_len, x96, y96, is_pick_chunks=False),
        output_signature=(sig_x, sig_y)
    )
    return tf.data.Dataset.sample_from_datasets([ds_pu, ds_fu], weights=[PICK_CHUNK_RATIO, FINGER_CHUNK_RATIO], seed=SEED)\
             .batch(batch_size).prefetch(tf.data.AUTOTUNE)

def approx_steps_from_length(total_samples, seg_len, batch_size, hop=None, cap=None):
    """
    Steps/epoch estimate from a single long stream length.
    Uses hop=seg_len//2 by default and an optional cap on steps.
    """
    hop = hop or max(1, seg_len // 2)
    windows = max((int(total_samples) - int(seg_len)) // int(hop) + 1, 1)
    steps = max(windows // int(batch_size), 1)
    if cap is not None:
        steps = min(steps, int(cap))
    return steps

def _fit_len(seg, seg_len):
    if len(seg) < seg_len:
        return np.pad(seg, (0, seg_len - len(seg)))
    return seg[:seg_len]

def segment_generator(seg_len, chunks, is_pick_chunks, model_sample_rate, pickup_flag=PICKUP_PIEZO):
    """
    Yields (x, (y_audio, style_flag, pickup_flag)) where x and y_audio are (seg_len, 1) at
    model_sample_rate. Slices whole-file views at the model rate (see _mono_at_sr), which avoids
    per-segment resample edge transients.
    """
    style_flag = 1.0 if is_pick_chunks else 0.0
    pickup_flag = float(pickup_flag)
    while True:
        src, tgt, cstart, cdur = random.choice(chunks)
        file_sr = _file_sr(src)

        x_all = _mono_at_sr(src, model_sample_rate)
        y_all = _mono_at_sr(tgt, model_sample_rate)

        # Chunk bounds converted from file-rate to model-rate samples
        ms = (cstart * int(model_sample_rate)) // file_sr
        md = (cdur   * int(model_sample_rate)) // file_sr
        if md <= 0:
            continue

        off = random.randrange(max(md - seg_len, 1))
        s = ms + off
        xs = x_all[s:s + seg_len]
        ys = y_all[s:s + seg_len]

        if xs.size == 0 or np.max(np.abs(xs)) < SEGMENT_GENERATOR_MIN_AMP_THRESH:
            continue

        xs = _fit_len(xs, seg_len)
        ys = _fit_len(ys, seg_len)

        yield xs.astype('float32')[:, None], \
              (ys.astype('float32')[:, None], np.float32(style_flag), np.float32(pickup_flag))

def segment_generator_dual(seg_len_realtime,
                           chunks,
                           is_pick_chunks,
                           realtime_sample_rate=REALTIME_SAMPLE_RATE,
                           offline_sample_rate=OFFLINE_SAMPLE_RATE,
                           pickup_flag=PICKUP_PIEZO):
    """
    Yields ((x_realtime, x_offline), (y_audio_realtime, style_flag, pickup_flag)).

    Supports any integer ratio offline_sample_rate : realtime_sample_rate. The realtime and
    offline segments are sliced from whole-file views at each rate with exactly aligned
    starts (s_offline = factor * s_realtime).
    """
    if offline_sample_rate % realtime_sample_rate != 0:
        raise ValueError(f"offline_sample_rate ({offline_sample_rate}) must be an integer multiple of realtime_sample_rate ({realtime_sample_rate}).")

    factor = offline_sample_rate // realtime_sample_rate
    seg_len_offline = factor * seg_len_realtime
    style_flag = 1.0 if is_pick_chunks else 0.0
    pickup_flag = float(pickup_flag)

    while True:
        src, tgt, cstart, cdur = random.choice(chunks)
        file_sr = _file_sr(src)

        x48 = _mono_at_sr(src, realtime_sample_rate)
        y48 = _mono_at_sr(tgt, realtime_sample_rate)
        x96 = _mono_at_sr(src, offline_sample_rate)

        ms48 = (cstart * int(realtime_sample_rate)) // file_sr
        md48 = (cdur   * int(realtime_sample_rate)) // file_sr
        if md48 <= 0:
            continue

        off48 = random.randrange(max(md48 - seg_len_realtime, 1))
        s48 = ms48 + off48
        s96 = s48 * factor

        xs48 = x48[s48:s48 + seg_len_realtime]
        ys48 = y48[s48:s48 + seg_len_realtime]
        # Teacher copy: the same span at 96k plus the (non-causal) teacher's own left/right context
        xs96 = _slice_zero_padded(x96, s96 - KD_TEACHER_LEFT_CTX_SAMPLES,
                                  s96 + seg_len_offline + KD_TEACHER_RIGHT_CTX_SAMPLES)

        if xs48.size == 0 or np.max(np.abs(xs48)) < SEGMENT_GENERATOR_MIN_AMP_THRESH:
            continue

        xs48 = _fit_len(xs48, seg_len_realtime)
        ys48 = _fit_len(ys48, seg_len_realtime)

        yield (xs48.astype('float32')[:, None], xs96.astype('float32')[:, None]), \
              (ys48.astype('float32')[:, None], np.float32(style_flag), np.float32(pickup_flag))

def approx_steps_from_chunks(chunks, seg_len, model_sample_rate, batch_size):
    """
    Estimate steps/epoch given chunks defined in file-rate samples while model uses model_sample_rate.
    Uses 50% overlap.
    """
    total_windows = 0
    for src, _, _, cdur in chunks:
        file_sr = _file_sr(src)
        need_src_len = int(math.ceil(seg_len * file_sr / float(model_sample_rate)))
        hop_src = max(1, need_src_len // 2)
        num = max((cdur - need_src_len) // hop_src + 1, 0)
        total_windows += num
    return 10 if DEBUG else max(total_windows // batch_size, 1)

# ────────────────────────────── TRAINING ─────────────────────────────────────

@register_keras_serializable()
def add_shared_random_gain_offline(db_lo=-24.0, db_hi=+12.0):
    """
    Apply the same random gain (in dB) to the offline student input and the mic target.
    Because the teacher is embedded in the finetune model and reuses this same input tensor,
    it automatically sees the same level, making mask KD invariant to absolute RMS.
    """
    def _map(inputs, outputs, sw):
        b = tf.shape(inputs['offline_piezo_input'])[0]
        g_db = tf.random.uniform((b, 1, 1), db_lo, db_hi, dtype=tf.float32)  # independent gain per sample
        g = tf.pow(10.0, g_db / 20.0)

        # Student input (teacher reuses this)
        inputs['offline_piezo_input'] = inputs['offline_piezo_input'] * g

        # Scale only the audio target; style target is dummy and unused
        y_audio, y_style = outputs
        outputs = (y_audio * g, y_style)
        return inputs, outputs, sw
    return _map

@register_keras_serializable()
def add_shared_random_gain_realtime(db_lo=-24.0, db_hi=+12.0):
    """
    Apply the same random gain (in dB) to the realtime student input and the mic target
    (single-rate variant, used when no teacher input is present).
    """
    def _map(inputs, outputs, sw):
        b = tf.shape(inputs['realtime_piezo_input'])[0]
        g_db = tf.random.uniform((b, 1, 1), db_lo, db_hi, dtype=tf.float32)  # independent gain per sample
        g = tf.pow(10.0, g_db / 20.0)
        inputs['realtime_piezo_input'] = inputs['realtime_piezo_input'] * g
        y_audio, y_style = outputs
        outputs = (y_audio * g, y_style)
        return inputs, outputs, sw
    return _map

@register_keras_serializable()
def add_shared_random_gain_dual(db_lo=-24.0, db_hi=+12.0):
    """
    Apply the same random gain (in dB) to:
      - realtime_piezo_input (student, 48k)
      - realtime_offline_piezo_input (teacher, 96k)
      - y_audio target (mic)
    This makes mask KD invariant to absolute input level.
    """
    def _map(inputs, outputs, sw):
        b = tf.shape(inputs['realtime_piezo_input'])[0]
        g_db = tf.random.uniform((b, 1, 1), db_lo, db_hi, dtype=tf.float32)  # independent gain per sample
        g = tf.pow(10.0, g_db / 20.0)

        xr  = inputs['realtime_piezo_input']
        xrt = inputs.get('realtime_offline_piezo_input', None)

        inputs['realtime_piezo_input'] = xr * g
        if xrt is not None:
            inputs['realtime_offline_piezo_input'] = xrt * g

        y_audio, y_style = outputs
        outputs = (y_audio * g, y_style)
        return inputs, outputs, sw
    return _map

@register_keras_serializable()
def add_sw_offline(x, y):
    y_audio, style_flag, pickup_flag = y
    b = tf.shape(x)[0];
    T = tf.shape(x)[1]
    style_true  = tf.expand_dims(style_flag, axis=-1)
    pickup      = tf.expand_dims(pickup_flag, axis=-1)
    force_mask = tf.cast(tf.random.uniform((b,1)) < 0.5, tf.float32)
    style_force = force_mask

    inputs = prepare_training_knob_feeds(
        kind="offline",
        piezo_input=x,
        style_true=style_true,
        style_force=style_force,
        pickup=pickup
    )

    outputs = (y_audio, style_true)
    sw_audio = tf.ones((b, T, 1), dtype=tf.float32)
    sw_style = tf.ones_like(style_true, dtype=tf.float32)
    return inputs, outputs, (sw_audio, sw_style)

@register_keras_serializable()
def add_sw_realtime(x, y):
    y_audio, style_flag, pickup_flag = y
    b = tf.shape(x)[0];
    T = tf.shape(x)[1]
    style_true  = tf.expand_dims(style_flag, axis=-1)
    pickup      = tf.expand_dims(pickup_flag, axis=-1)
    force_mask = tf.cast(tf.random.uniform((b,1)) < 0.5, tf.float32)
    style_force = force_mask

    inputs = prepare_training_knob_feeds(
        kind="realtime",
        piezo_input=x,
        style_true=style_true,
        style_force=style_force,
        pickup=pickup
    )

    outputs = (y_audio, style_true)
    sw_audio = tf.ones((b, T, 1), dtype=tf.float32)
    sw_style = tf.ones_like(style_true, dtype=tf.float32)
    return inputs, outputs, (sw_audio, sw_style)

@register_keras_serializable()
def add_sw_realtime_dual_nf(x, y, pickup=PICKUP_PIEZO):
    x_realtime, x_offline = x
    y_audio, _ = y
    b = tf.shape(x_realtime)[0]
    T_realtime = tf.shape(x_realtime)[1]
    style_true  = tf.zeros((b, 1), dtype=tf.float32)
    style_force = tf.zeros((b, 1), dtype=tf.float32)
    
    base_inputs = prepare_training_knob_feeds(
        kind="realtime",
        piezo_input=x_realtime,
        style_true=style_true,
        style_force=style_force,
        pickup=tf.ones((b, 1), tf.float32) * float(pickup)
    )
    
    inputs = {
        **base_inputs,
        'realtime_offline_piezo_input': x_offline
    }
    
    outputs = (y_audio, style_true)
    sw_audio = tf.ones((b, T_realtime, 1), dtype=tf.float32)
    sw_style = tf.zeros_like(style_true, dtype=tf.float32)
    return inputs, outputs, (sw_audio, sw_style)


@register_keras_serializable()
def add_sw_realtime_dual(x, y):
    x_realtime, x_offline = x
    y_audio, style_flag, pickup_flag = y
    b = tf.shape(x_realtime)[0];
    T_realtime = tf.shape(x_realtime)[1]
    style_true  = tf.expand_dims(style_flag, axis=-1)
    pickup      = tf.expand_dims(pickup_flag, axis=-1)
    force_mask = tf.cast(tf.random.uniform((b,1)) < 0.5, tf.float32)
    style_force = force_mask

    base_inputs = prepare_training_knob_feeds(
        kind="realtime",
        piezo_input=x_realtime,
        style_true=style_true,
        style_force=style_force,
        pickup=pickup
    )
    
    inputs = {
        **base_inputs,
        'realtime_offline_piezo_input': x_offline
    }
    
    outputs = (y_audio, style_true)
    sw_audio = tf.ones((b, T_realtime, 1), dtype=tf.float32)
    sw_style = tf.ones_like(style_true, dtype=tf.float32)
    return inputs, outputs, (sw_audio, sw_style)


@register_keras_serializable()
def add_sw_offline_finetune_nf(x, y, pickup=PICKUP_PIEZO):
    y_audio, _ = y
    b = tf.shape(x)[0]
    T = tf.shape(x)[1]

    style_true  = tf.zeros((b, 1), dtype=tf.float32)
    style_force = tf.zeros((b, 1), dtype=tf.float32)
    
    inputs = prepare_training_knob_feeds(
        kind="offline",
        piezo_input=x,
        style_true=style_true,
        style_force=style_force,
        pickup=tf.ones((b, 1), tf.float32) * float(pickup)
    )

    outputs = (y_audio, style_true)
    sw_audio = tf.ones((b, T, 1), dtype=tf.float32)
    sw_style = tf.zeros_like(style_true, dtype=tf.float32)
    return inputs, outputs, (sw_audio, sw_style)

def _spectral_denoise_target(y, sr,
                             hop_ms=FINETUNE_TARGET_DENOISE_HOP_MS,
                             beta=FINETUNE_TARGET_DENOISE_BETA,
                             floor_db=FINETUNE_TARGET_DENOISE_FLOOR_DB,
                             noise_q=FINETUNE_TARGET_DENOISE_NOISE_Q,
                             min_noise_frames=FINETUNE_TARGET_DENOISE_MIN_FRAMES):
    """
    Light spectral denoiser that is COLA/NOLA-safe and compatible with older SciPy
    (no 'length' kw on istft). Falls back to manual trim/pad.
    """

    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1 or y.size < 8:
        return y

    # Enforce Hann + 50% overlap → COLA/NOLA satisfied
    hop  = max(16, int(round((hop_ms / 1000.0) * sr)))
    nper = max(64, 2 * hop)            # 50% overlap for Hann
    if (nper % 2) == 1:
        nper += 1
    nover = nper - hop
    nfft = nper                        # no need for extra zero-pad

    # STFT with zero boundary and pad to keep edges NOLA-safe
    f, t, Z = spsig.stft(
        y, fs=sr,
        window='hann',
        nperseg=nper,
        noverlap=nover,
        nfft=nfft,
        boundary='zeros',
        padded=True
    )

    mag = np.abs(Z)
    frame_energy = np.sqrt(np.mean(mag * mag, axis=0) + 1e-12)
    thr = np.quantile(frame_energy, float(noise_q))
    idx = np.where(frame_energy <= thr)[0]
    if idx.size >= int(min_noise_frames):
        noise_profile = np.median(mag[:, idx], axis=1)
    else:
        noise_profile = np.percentile(mag, 20.0, axis=1)

    eps = 1e-8
    mag2 = mag * mag
    n2   = (noise_profile[:, None] ** 2)
    G = mag2 / (mag2 + float(beta) * n2 + eps)  # Wiener-like mask
    floor = 10.0 ** (float(floor_db) / 20.0)
    G = np.clip(G, floor, 1.0)

    Z_d = G * Z

    # Older SciPy has no 'length' kw; try it, then fallback and trim manually
    try:
        _, y_rec = spsig.istft(
            Z_d, fs=sr,
            window='hann',
            nperseg=nper,
            noverlap=nover,
            nfft=nfft,
            input_onesided=True,
            length=y.size  # SciPy >= 1.10
        )
    except TypeError:
        _, y_rec = spsig.istft(
            Z_d, fs=sr,
            window='hann',
            nperseg=nper,
            noverlap=nover,
            nfft=nfft,
            input_onesided=True
        )
        # Trim/pad to original length
        if y_rec.size > y.size:
            y_rec = y_rec[:y.size]
        elif y_rec.size < y.size:
            y_rec = np.pad(y_rec, (0, y.size - y_rec.size))

    return y_rec.astype(np.float32, copy=False)

def denoise_target_if_enabled(y_full, sr):
    """
    Apply the light spectral denoiser to the mic target channel(s) if enabled.
    y_full: (N, C) float32
    """
    if not FINETUNE_TARGET_DENOISE_ENABLE:
        return y_full
    if y_full.ndim != 2 or y_full.shape[0] < 8:
        return y_full
    Y = y_full.copy()
    for c in range(Y.shape[1]):
        Y[:, c] = _spectral_denoise_target(Y[:, c], sr)
    print("[finetune] Applied light spectral denoise to mic target.")
    return Y

def cleanup_and_prepare_train (logs_out_dir, checkpoints_out_dir, demo_checkpoints_out_dir):
    for dir_path in [logs_out_dir, demo_checkpoints_out_dir]:
        for sub in os.listdir(dir_path):
            sub_path = os.path.join(dir_path, sub)
            if os.path.isfile(sub_path):
                os.remove(sub_path)
            elif os.path.isdir(sub_path):
                shutil.rmtree(sub_path)
    # Sort numerically by epoch (a plain lexicographic sort breaks past epoch 99)
    ckpts = sorted(glob.glob(os.path.join(checkpoints_out_dir, "*.keras")),
                   key=_extract_epoch_from_ckpt)
    resume = len(ckpts) > 0
    resume_ckpt = ckpts[-1] if resume else None
    if resume:
        ep = _extract_epoch_from_ckpt(resume_ckpt)
        print(f"→ Found checkpoint, will resume from epoch {ep}: {resume_ckpt}")
        return (resume, resume_ckpt)
    else:
        return (False, None)
