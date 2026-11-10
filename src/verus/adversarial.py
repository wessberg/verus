import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable  # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import (Input, Layer)  # pyright: ignore[reportMissingImports]

from .config import (
    DISC_CHANNELS,
    DISC_LR,
    DISC_MIN_HZ,
    DISC_RESOLUTIONS_OFFLINE,
    OFFLINE_ADV_RAMP_EPOCHS,
    OFFLINE_ADV_START_EPOCH,
    OFFLINE_SAMPLE_RATE,
    ULTRALOW_MAIN_CUTOFF_HZ,
)
from .dsp import (
    _crop_time,
    _make_lpf_kernel,
)

# ─── Adversarial training (offline model) ─────────────────────────────────────
# The mic's pick-attack HF is not phase-predictable from the piezo, so no reconstruction loss can
# ask for it without asking for its (unknowable) waveform. A discriminator on log-magnitude
# spectrograms instead asks for HF that is PLAUSIBLE — right spectral envelope, right time course,
# noise-like where the target is noise-like — and feature matching ties it to the paired target.

@register_keras_serializable()
class LogMagSpectrogram(Layer):
    """(B,T,1) audio → (B, frames, bins, 1) log-magnitude STFT of the bins >= min_hz."""
    def __init__(self, n_fft, hop, sample_rate=OFFLINE_SAMPLE_RATE, min_hz=0.0, **kwargs):
        super().__init__(**kwargs)
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.sample_rate = int(sample_rate)
        self.min_hz = float(min_hz)

    def call(self, x):
        s = tf.squeeze(x, axis=-1)
        w = tf.signal.hann_window(self.n_fft, periodic=True, dtype=s.dtype)
        m = tf.abs(tf.signal.rfft(tf.signal.frame(s, self.n_fft, self.hop, pad_end=True) * w,
                                  fft_length=[self.n_fft]))
        k0 = min(self.n_fft // 2, int(np.ceil(self.min_hz / (float(self.sample_rate) / self.n_fft))))
        return tf.math.log(m[..., k0:] + 1e-5)[..., None]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_fft": self.n_fft, "hop": self.hop, "sample_rate": self.sample_rate, "min_hz": self.min_hz})
        return cfg

def build_stft_discriminator(n_fft, hop, sample_rate=OFFLINE_SAMPLE_RATE, min_hz=DISC_MIN_HZ,
                             channels=DISC_CHANNELS, name=None):
    """
    One resolution of the multi-resolution spectrogram discriminator (UnivNet/EnCodec style,
    magnitude only). Outputs [feature maps..., logits] — the features feed the feature-matching loss.
    """
    name = name or f"disc_stft_{n_fft}"
    inp = Input(shape=(None, 1), name=f"{name}_audio")
    x = LogMagSpectrogram(n_fft, hop, sample_rate=sample_rate, min_hz=min_hz, name=f"{name}_spec")(inp)
    feats = []
    x = tf.keras.layers.Conv2D(channels, (3, 9), padding="same", name=f"{name}_c0")(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.2, name=f"{name}_a0")(x)
    feats.append(x)
    for i in range(1, 4):
        x = tf.keras.layers.Conv2D(channels, (3, 9), strides=(1, 2), padding="same", name=f"{name}_c{i}")(x)
        x = tf.keras.layers.LeakyReLU(negative_slope=0.2, name=f"{name}_a{i}")(x)
        feats.append(x)
    x = tf.keras.layers.Conv2D(channels, (3, 3), padding="same", name=f"{name}_c4")(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.2, name=f"{name}_a4")(x)
    feats.append(x)
    logits = tf.keras.layers.Conv2D(1, (3, 3), padding="same", name=f"{name}_logits")(x)
    return tf.keras.Model(inp, feats + [logits], name=name)

def build_offline_discriminators():
    return [build_stft_discriminator(n, h, sample_rate=OFFLINE_SAMPLE_RATE) for (n, h) in DISC_RESOLUTIONS_OFFLINE]

def disc_weights_path(ckpt_path, i):
    return f"{os.path.splitext(ckpt_path)[0]}.disc{i}.weights.h5"

class AdversarialTrainer(tf.keras.Model):
    """
    Trains `generator` (the offline training graph) with its reconstruction loss + style BCE + its
    own add_loss terms, PLUS an LSGAN adversarial loss and feature matching against `discriminators`,
    which are updated in the same step. Only the loss core is judged. fit()/evaluate() run on this
    wrapper; save the GENERATOR (GeneratorCheckpoint does), never this wrapper.
    val_loss = reconstruction + style (+ regularisers), i.e. comparable to non-adversarial runs.
    """
    def __init__(self, generator, discriminators, recon_loss, style_weight, adv_weight, fm_weight,
                 core, d_learning_rate=DISC_LR, **kwargs):
        super().__init__(name=kwargs.pop("name", "adversarial_trainer"), **kwargs)
        self.generator = generator
        self.discriminators = list(discriminators)
        self.recon_loss = recon_loss
        self.style_weight = float(style_weight)
        self.adv_weight = float(adv_weight)
        self.fm_weight = float(fm_weight)
        self.core = core
        self.adv_scale = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="adv_scale")
        self.d_optimizer = tf.keras.optimizers.Adam(d_learning_rate, beta_1=0.5, beta_2=0.9, clipnorm=10.0)
        self._trackers = {k: tf.keras.metrics.Mean(name=k) for k in
                          ("loss", "recon", "style", "adv", "fm", "d_loss", "d_real", "d_fake")}

    @property
    def metrics(self):
        return list(self._trackers.values())

    def call(self, inputs, training=False):
        return self.generator(inputs, training=training)

    def _disc(self, y):
        return [d(y, training=True) for d in self.discriminators]

    def _gen_losses(self, x, y_audio, y_style, training):
        y_pred, s_pred = self.generator(x, training=training)
        recon = self.recon_loss(y_audio, y_pred)
        style = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_style, s_pred))
        reg = tf.add_n(self.generator.losses) if self.generator.losses else tf.constant(0.0)
        return y_pred, recon, style, reg

    def train_step(self, data):
        x, y, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_audio, y_style = y
        real = _crop_time(y_audio, *self.core)
        with tf.GradientTape() as g_tape:
            y_pred, recon, style, reg = self._gen_losses(x, y_audio, y_style, training=True)
            fake = _crop_time(y_pred, *self.core)
            d_fake = self._disc(fake)
            d_real = [[tf.stop_gradient(t) for t in outs] for outs in self._disc(real)]
            adv = tf.add_n([tf.reduce_mean(tf.square(1.0 - o[-1])) for o in d_fake])
            fm = tf.add_n([tf.add_n([tf.reduce_mean(tf.abs(r - f)) for r, f in zip(ro[:-1], fo[:-1])]) / float(len(fo) - 1)
                           for ro, fo in zip(d_real, d_fake)])
            g_total = recon + self.style_weight * style + reg + self.adv_scale * (self.adv_weight * adv + self.fm_weight * fm)
        g_vars = self.generator.trainable_variables
        self.optimizer.apply_gradients(zip(g_tape.gradient(g_total, g_vars), g_vars))

        with tf.GradientTape() as d_tape:
            lr_ = [o[-1] for o in self._disc(real)]
            lf_ = [o[-1] for o in self._disc(tf.stop_gradient(fake))]
            d_loss = tf.add_n([tf.reduce_mean(tf.square(1.0 - r)) + tf.reduce_mean(tf.square(f)) for r, f in zip(lr_, lf_)])
        d_vars = [v for d in self.discriminators for v in d.trainable_variables]
        self.d_optimizer.apply_gradients(zip(d_tape.gradient(d_loss, d_vars), d_vars))

        vals = {"loss": recon + self.style_weight * style + reg, "recon": recon, "style": style, "adv": adv,
                "fm": fm, "d_loss": d_loss,
                "d_real": tf.add_n([tf.reduce_mean(r) for r in lr_]) / len(lr_),
                "d_fake": tf.add_n([tf.reduce_mean(f) for f in lf_]) / len(lf_)}
        for k, v in vals.items():
            self._trackers[k].update_state(v)
        return {k: t.result() for k, t in self._trackers.items()}

    def test_step(self, data):
        x, y, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_audio, y_style = y
        _, recon, style, reg = self._gen_losses(x, y_audio, y_style, training=False)
        for k, v in (("loss", recon + self.style_weight * style + reg), ("recon", recon), ("style", style)):
            self._trackers[k].update_state(v)
        return {k: self._trackers[k].result() for k in ("loss", "recon", "style")}

class GeneratorCheckpoint(tf.keras.callbacks.Callback):
    """ModelCheckpoint for AdversarialTrainer: saves the generator (a normal .keras model, loadable by
    the resume logic and --promote-offline-checkpoint) and the discriminator weights next to it.
    Saves EVERY epoch by default: the reconstruction val_loss is not the criterion an adversarial
    run should be selected by (the adversarial term trades a little reconstruction for plausible HF),
    so pick the epoch with --analyze-attacks / by ear."""
    def __init__(self, filepath, monitor="val_loss", save_best_only=False):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = bool(save_best_only)
        self.best = np.inf

    def on_epoch_end(self, epoch, logs=None):
        cur = (logs or {}).get(self.monitor)
        if cur is None or not np.isfinite(cur):
            return
        improved = cur < self.best
        if self.save_best_only and not improved:
            print(f"\nEpoch {epoch + 1}: {self.monitor} did not improve from {self.best:.5f}")
            return
        path = self.filepath.format(epoch=epoch + 1, **{k: float(v) for k, v in (logs or {}).items()})
        print(f"\nEpoch {epoch + 1}: {self.monitor} {cur:.5f}" + (" (best so far)" if improved else "")
              + f", saving generator to {path}")
        self.best = min(self.best, float(cur))
        self.model.generator.save(path)
        for i, d in enumerate(self.model.discriminators):
            d.save_weights(disc_weights_path(path, i))

class AdversarialRamp(tf.keras.callbacks.Callback):
    """Generator's adversarial weight: 0 before start_epoch, then a linear ramp to 1 over ramp_epochs."""
    def __init__(self, start_epoch=OFFLINE_ADV_START_EPOCH, ramp_epochs=OFFLINE_ADV_RAMP_EPOCHS):
        super().__init__()
        self.start_epoch = int(start_epoch)
        self.ramp_epochs = max(1, int(ramp_epochs))

    def on_epoch_begin(self, epoch, logs=None):
        s = 0.0 if epoch < self.start_epoch else min(1.0, (epoch - self.start_epoch + 1) / float(self.ramp_epochs))
        self.model.adv_scale.assign(s)
        print(f"[adversarial] epoch {epoch + 1}: generator adversarial scale {s:.2f}")

@register_keras_serializable()
def lowband_error_penalty(y_true, y_pred, sample_rate,
                          cutoff_hz=ULTRALOW_MAIN_CUTOFF_HZ,
                          order=129,
                          window="hamming"):
    """
    Training loss term: L1 of LPF(y_pred - y_true) below cutoff_hz.
    Returns (B,T,1) so it’s compatible with per-sample weighting.
    """
    err = y_pred - y_true
    k = _make_lpf_kernel(int(sample_rate), float(cutoff_hz), int(order), window=window)  # (L,1,1) reversed
    pad = tf.shape(k)[0] - 1
    err_pad = tf.pad(err, [[0,0],[pad,0],[0,0]])
    lp_err = tf.nn.conv1d(err_pad, filters=k, stride=1, padding='VALID')
    return tf.abs(lp_err)
