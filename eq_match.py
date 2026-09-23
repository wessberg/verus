"""
Match the long-term tone of the acoustic's magnetic (soundhole) pickup to an electric guitar's passive
pickup, and write the correction as a minimum-phase FIR. It turns the magnetic triplet recordings into
"fake electric" training sources (BACKLOG B-6): an electric can't be recorded playing the same
performance as the acoustic's mic, so the match is between the long-term average spectra of two
recordings of the same material, one per guitar. How to record them: TRAINING.md §4.

    python eq_match.py input/eqmatch/acoustic_magnetic.wav input/eqmatch/electric_humbucker.wav

Record both through the same interface input and gain, unprocessed, >= 5 min of the same material in
the same style, with a few seconds of silence (strings muted) at the start and end: the quietest
frames give each file's noise floor (hum, hiss), which is subtracted before matching.
input/eqmatch/ is not read by training.

Writes to --out (default out/eqmatch):
  eq_curve.csv                electric/acoustic correction on a log grid: freq_hz, gain_db, measured
                              (1 = inside the band where both files are well above their noise floor)
  electric_match_ir_96k.wav   the correction as a minimum-phase FIR at 96 and 48 kHz (32-bit float;
  electric_match_ir_48k.wav   also loads in any IR/convolution plugin)
  acoustic_as_electric.wav    the acoustic recording through the FIR, at the electric's playing level,
                              to compare by ear with the electric DI (32-bit float)
  eq_match.png                spectra and correction (when matplotlib is installed)

The FIR is level-neutral on the acoustic recording (training applies random gain anyway). Outside the
measured band the correction is held (bass) or continued at the last octave's slope (treble).
Audio is streamed in blocks, never held whole in memory.
"""
import argparse, csv, os, sys

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

GRID_PER_OCT = 48
GRID_F0, GRID_F1 = 10.0, 48000.0   # the grid covers both FIR rates' Nyquist


def long_term_spectrum(path, nfft, gate_db, noise_sec):
    """Mean power spectrum of the playing frames and of the quietest frames (channel 0)."""
    info = sf.info(path)
    sr, hop = int(info.samplerate), nfft // 4
    w = np.hanning(nfft)

    def frames():
        for blk in sf.blocks(path, blocksize=nfft, overlap=nfft - hop, dtype="float32", always_2d=True):
            if blk.shape[0] < nfft:
                return
            yield blk[:, 0].astype(np.float64)

    energies, clipped = [], 0
    for x in frames():                                   # pass 1: frame energies for the gate
        clipped += int(np.count_nonzero(np.abs(x[:hop]) >= 0.999))
        energies.append(float(np.sum((w * x) ** 2)))
    e = np.asarray(energies)
    nz = e > 0                                           # digital silence carries no noise information
    if np.count_nonzero(nz) < 32:
        sys.exit(f"{path}: too short (need at least a few seconds of audio)")
    e_db = np.full(e.shape, -np.inf)
    e_db[nz] = 10 * np.log10(e[nz])

    ref = np.percentile(e_db[nz], 95)
    n_noise = max(4, int(noise_sec * sr / hop))
    order = np.flatnonzero(nz)[np.argsort(e_db[nz])]
    noise = np.zeros(e.shape, bool)
    noise[order[:n_noise]] = True
    noise_db = float(np.median(e_db[noise]))
    has_silence = noise_db < ref - 35                    # a decaying note is not a noise floor
    play = e_db > max(ref - gate_db, noise_db + 15)

    S = np.zeros((2, nfft // 2 + 1))                     # first and second half of the recording
    N = np.zeros(nfft // 2 + 1)
    for i, x in enumerate(frames()):                     # pass 2: spectra of the selected frames
        if play[i] or noise[i]:
            P = np.abs(np.fft.rfft(w * x)) ** 2
            if play[i]:
                S[int(i >= len(e) // 2)] += P
            if noise[i]:
                N += P
    n_half = np.array([np.count_nonzero(play[:len(e) // 2]), np.count_nonzero(play[len(e) // 2:])])
    halves = S / np.maximum(n_half, 1)[:, None]
    S = S.sum(axis=0) / np.count_nonzero(play)
    N = N / np.count_nonzero(noise) if has_silence else np.zeros_like(N)
    return dict(path=path, sr=sr, freqs=np.fft.rfftfreq(nfft, 1 / sr), S=S, N=N, halves=halves,
                seconds=info.frames / sr, channels=info.channels, n_play=int(np.count_nonzero(play)),
                play_sec=np.count_nonzero(play) * hop / sr, snr_db=ref - noise_db,
                has_silence=has_silence, clipped=clipped)


def smooth_to_grid(freqs, P, grid, frac):
    """1/frac-octave band average of the power spectrum P at each grid frequency."""
    c = np.concatenate([[0.0], np.cumsum(P)])
    lo = np.searchsorted(freqs, grid * 2 ** (-0.5 / frac))
    hi = np.searchsorted(freqs, grid * 2 ** (0.5 / frac), side="right")
    hi = np.clip(hi, 1, len(P))
    lo = np.minimum(lo, hi - 1)
    return (c[hi] - c[lo]) / (hi - lo)


def interp_db(grid, gain_db, f):
    """Correction at arbitrary frequencies (log-frequency interpolation, held beyond the grid)."""
    return np.interp(np.log(np.maximum(f, grid[0])), np.log(grid), gain_db)


def min_phase_fir(grid, gain_db, sr, ntaps):
    """Minimum-phase FIR with magnitude gain_db (real-cepstrum folding), faded over its last quarter."""
    nfft = 1 << int(np.ceil(np.log2(16 * ntaps)))
    logmag = interp_db(grid, gain_db, np.fft.rfftfreq(nfft, 1 / sr)) * (np.log(10) / 20)
    cep = np.fft.irfft(logmag, nfft)
    fold = np.zeros(nfft)
    fold[0], fold[nfft // 2] = cep[0], cep[nfft // 2]
    fold[1:nfft // 2] = 2 * cep[1:nfft // 2]
    h = np.fft.irfft(np.exp(np.fft.rfft(fold)), nfft)
    kept = h[:ntaps].copy()
    nf = ntaps // 4
    kept[-nf:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, nf)))
    return kept, 1 - np.sum(kept ** 2) / np.sum(h ** 2)


def fir_response_db(h, sr, f):
    nfft = 1 << int(np.ceil(np.log2(16 * len(h))))
    H = 20 * np.log10(np.abs(np.fft.rfft(h, nfft)) + 1e-12)
    return np.interp(f, np.fft.rfftfreq(nfft, 1 / sr), H)


def convolve_file(src, dst, h, gain):
    """Block overlap-add convolution of src (channel 0) with h, written as 32-bit float."""
    info = sf.info(src)
    tail = np.zeros(len(h) - 1)
    with sf.SoundFile(dst, "w", info.samplerate, 1, subtype="FLOAT") as out:
        for blk in sf.blocks(src, blocksize=1 << 18, dtype="float32", always_2d=True):
            y = fftconvolve(blk[:, 0].astype(np.float64), h) * gain
            y[:len(tail)] += tail
            n = blk.shape[0]
            out.write(y[:n].astype(np.float32))
            tail = y[n:]
        out.write(tail.astype(np.float32))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("acoustic", help="the acoustic's magnetic pickup recording")
    ap.add_argument("electric", help="the electric guitar DI, same material")
    ap.add_argument("--out", default=os.path.join("out", "eqmatch"))
    ap.add_argument("--smooth", type=float, default=3, help="smoothing, 1/N octave (default 3)")
    ap.add_argument("--taps", type=int, default=8192, help="FIR length at 96 kHz (48 kHz: half)")
    ap.add_argument("--nfft", type=int, default=16384, help="analysis frame length in samples")
    ap.add_argument("--gate-db", type=float, default=30, help="use frames within this of the loud frames")
    ap.add_argument("--snr-db", type=float, default=10, help="measured band: both this far above noise")
    ap.add_argument("--noise-sec", type=float, default=2, help="seconds of quietest frames = noise floor")
    ap.add_argument("--fmin", type=float, default=70, help="held flat below (the lowest string is 82 Hz)")
    ap.add_argument("--fmax", type=float, default=20000)
    ap.add_argument("--no-audition", action="store_true", help="skip writing acoustic_as_electric.wav")
    a = ap.parse_args()

    spec = {}
    for name, path in (("acoustic", a.acoustic), ("electric", a.electric)):
        s = spec[name] = long_term_spectrum(path, a.nfft, a.gate_db, a.noise_sec)
        print(f"{name}: {s['seconds']:.1f} s at {s['sr']} Hz, {s['play_sec']:.1f} s of playing used, "
              f"playing {s['snr_db']:.0f} dB above the quietest {a.noise_sec:g} s")
        if s["channels"] > 1:
            print(f"  note: {s['channels']} channels, only channel 1 is used")
        if not s["has_silence"]:
            print("  warning: no silence found (quietest frames are within 35 dB of the playing) - "
                  "noise floor not subtracted; leave a few seconds of muted strings at the start/end")
        if s["clipped"]:
            print(f"  warning: {s['clipped']} samples at full scale - re-record with less gain")

    grid = GRID_F0 * 2 ** (np.arange(int(np.log2(GRID_F1 / GRID_F0) * GRID_PER_OCT) + 1) / GRID_PER_OCT)
    sm, clean, ok = {}, {}, np.ones(grid.shape, bool)
    for name, s in spec.items():
        S = smooth_to_grid(s["freqs"], s["S"], grid, a.smooth)
        N = smooth_to_grid(s["freqs"], s["N"], grid, a.smooth)
        sm[name] = (S, N)
        clean[name] = np.maximum(S - N, 1e-30)
        ok &= (S > N * 10 ** (a.snr_db / 10)) & (grid < s["sr"] / 2 * 0.95)
    ok &= (grid >= a.fmin) & (grid <= a.fmax)
    i1k = int(np.argmin(np.abs(grid - 1000)))
    if not ok[i1k]:
        sys.exit("both recordings need to be well above their noise floor around 1 kHz")
    i_lo, i_hi = i1k, i1k                                # contiguous measured band around 1 kHz
    while i_lo > 0 and ok[i_lo - 1]:
        i_lo -= 1
    while i_hi < len(grid) - 1 and ok[i_hi + 1]:
        i_hi += 1
    measured = np.zeros(grid.shape, bool)
    measured[i_lo:i_hi + 1] = True

    gain = 10 * np.log10(clean["electric"] / clean["acoustic"])

    # Split-half check: the part of the curve that depends on which notes were played, not on the pickups
    # (measured on synthetic data: different 3-5 min passages put the curve 1.5-2 dB RMS off; 1 min, 4 dB)
    half_gain = []
    for k in (0, 1):
        c = {n: np.maximum(smooth_to_grid(s["freqs"], s["halves"][k], grid, a.smooth) - sm[n][1], 1e-30)
             for n, s in spec.items()}
        half_gain.append(10 * np.log10(c["electric"] / c["acoustic"]))
    gain[:i_lo] = gain[i_lo]
    top = grid >= grid[i_hi] / 2
    top &= measured
    slope = float(np.polyfit(np.log2(grid[top]), gain[top], 1)[0]) if np.count_nonzero(top) > 2 else 0.0
    slope = min(max(slope, -24.0), 0.0)                  # never extrapolate a treble boost
    beyond = np.arange(len(grid)) > i_hi
    gain[beyond] = np.maximum(gain[i_hi] + slope * np.log2(grid[beyond] / grid[i_hi]), gain[i_hi] - 48)

    # Level-neutral on the acoustic recording; the raw offset is the electric's relative playing level
    sa = spec["acoustic"]
    Pa = np.maximum(sa["S"] - sa["N"], 0)
    level_offset = 10 * np.log10(np.sum(Pa * 10 ** (interp_db(grid, gain, sa["freqs"]) / 10)) / np.sum(Pa))
    gain -= level_offset
    se = spec["electric"]
    level_db = 10 * np.log10(np.sum(np.maximum(se["S"] - se["N"], 0)) / np.sum(Pa))

    chk = measured & (grid >= 80) & (grid <= 12000)
    d = half_gain[0] - half_gain[1]
    d -= np.median(d[chk])
    rms_half = float(np.sqrt(np.mean(d[chk] ** 2)))
    print(f"split-half check: curves from the first and second halves differ by {rms_half:.1f} dB RMS "
          f"(max {np.max(np.abs(d[chk])):.1f} dB) over 80 Hz - 12 kHz; the full curve is off by roughly half that"
          + (" - record longer, or keep the material on both guitars more alike" if rms_half > 3 else ""))

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "eq_curve.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["freq_hz", "gain_db", "measured"])
        for fr, g, m in zip(grid, gain, measured):
            wr.writerow([f"{fr:.2f}", f"{g:.3f}", int(m)])

    firs = {}
    band = measured & (grid >= max(grid[i_lo], 60.0))
    for sr, ntaps in ((96000, a.taps), (48000, a.taps // 2)):
        h, lost = min_phase_fir(grid, gain, sr, ntaps)
        firs[sr] = h
        sf.write(os.path.join(a.out, f"electric_match_ir_{sr // 1000}k.wav"), h.astype(np.float32), sr,
                 subtype="FLOAT")
        err = fir_response_db(h, sr, grid[band]) - gain[band]
        print(f"FIR {ntaps} taps at {sr // 1000} kHz: within {np.max(np.abs(err)):.2f} dB of the curve over "
              f"{grid[band][0]:.0f} Hz - {grid[band][-1] / 1000:.1f} kHz (truncation dropped {lost:.1e} of its energy)")

    print(f"electric playing level: {level_db:+.1f} dB relative to the acoustic "
          f"(meaningful only with the same interface gain and similar dynamics)")
    print(f"measured band: {grid[i_lo]:.0f} Hz - {grid[i_hi] / 1000:.1f} kHz; "
          f"beyond it the treble continues at {slope:.1f} dB/octave")
    marks = [63, 125, 250, 500, 1000, 2000, 3000, 4000, 6000, 8000, 12000, 16000]
    print("correction  " + "".join(f"{(f'{m // 1000}k' if m >= 1000 else str(m)):>7}" for m in marks))
    print("        dB  " + "".join(f"{g:>+7.1f}" for g in interp_db(grid, gain, np.array(marks, float))))

    if not a.no_audition:
        dst = os.path.join(a.out, "acoustic_as_electric.wav")
        h = firs[sa["sr"]] if sa["sr"] in firs else min_phase_fir(grid, gain, sa["sr"], a.taps * sa["sr"] // 96000)[0]
        convolve_file(a.acoustic, dst, h, 10 ** (level_db / 20))
        print(f"wrote {dst}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
    if plt is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ref = 10 * np.log10(np.max(clean["acoustic"][measured]))
        for name, color in (("acoustic", "tab:blue"), ("electric", "tab:red")):
            off = ref + (level_db if name == "electric" else 0.0)    # electric shown at the acoustic's level
            ax1.semilogx(grid, 10 * np.log10(clean[name]) - off, color=color, label=name)
            ax1.semilogx(grid, 10 * np.log10(np.maximum(sm[name][1], 1e-30)) - off,
                         color=color, ls=":", lw=0.8, label=f"{name} noise floor")
        realized = fir_response_db(firs[96000], 96000, grid)
        ax1.semilogx(grid, 10 * np.log10(clean["acoustic"]) - ref + realized, color="k", lw=0.8,
                     label="acoustic through the FIR")
        ax1.set_ylim(-90, 10)
        ax1.set_ylabel("dB")
        ax1.legend(fontsize=8)
        ax1.grid(True, which="both", alpha=0.3)
        ax2.semilogx(grid, gain, label="correction curve")
        ax2.semilogx(grid, realized, ls="--", label="96 kHz FIR")
        ax2.axvspan(grid[i_lo], grid[i_hi], color="tab:green", alpha=0.08, label="measured band")
        ax2.set_ylim(max(-48, gain.min() - 3), gain.max() + 3)
        ax2.set_xlim(20, 24000)
        ax2.set_xlabel("Hz")
        ax2.set_ylabel("dB")
        ax2.legend(fontsize=8)
        ax2.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "eq_match.png"), dpi=110)
        print(f"wrote {os.path.join(a.out, 'eq_match.png')}")
    print(f"wrote {os.path.join(a.out, 'eq_curve.csv')} and the FIRs")


if __name__ == "__main__":
    main()
