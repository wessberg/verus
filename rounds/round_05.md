# Round 05 — Third review pass: KD alignment, polyphase upsamplers, a pedal that builds on the Pi

- **Date:** 2026-09-22
- **Agent:** Claude (Claude Code)
- **Goal:** a third full review and fix pass (`piezo2mic.py`, `cpp/src/main.cpp`), verified off-target.
- **Commits:** committed in round 07 together with rounds 06–07 ("Rounds 05-07: review fixes, pick-attack objective, memory-safe training, EQ match, rounds docs"), merged into `master`.

*Written retroactively in round 07 from the previous agent's notes and the old `TRAINING.md` §10.*

## What happened

Training:
- `_kd_align` used a 2-tap average pool: the KD target was half a 96 kHz sample late and 24–48 kHz
  aliased at about −5 dB. Now the `resample_poly` 2:1 filter centred on 2n (matches `resample_poly` to
  3e-7); the teacher crop keeps `KD_TEACHER_DECIM_MARGIN` = 20 samples on the right.
- KD layers added their losses in evaluation too, so `val_loss` (EarlyStopping, checkpoints, LR
  plateau) fell with the KD alpha schedule. Now training-only; `KDFeatureDistill.proj` built in `build()`.
- `KD_ALPHA_SCHEDULE` ended at epoch 80 with `EPOCHS` = 40 → `KD_ANNEAL_END_EPOCH = EPOCHS`. The style
  blend was trainable *and* scheduled (0.7 → 0.3 over 100 epochs) → non-trainable,
  `KD_STYLE_BLEND_SCHEDULE` 0.7 → 0 by 3/4 of `EPOCHS`.
- The "fixed" validation set re-drew its style-force mask every epoch (map after cache) → map before cache.
- Left contexts treated the colour EQ as a parallel branch (max); it filters the sum, so it adds:
  offline 19 416 → 19 512, realtime 8430 → 8494 (`_offline_train_left_context`,
  `_realtime_branch_left_contexts`).
- Fine-tune: `--pickup` (was hard-wired to piezo); a freshly fine-tuned offline model is the teacher;
  LR persisted across resume (`train_state.json`); realtime training no longer saves the KD graph over
  the streaming model path.

Model / export:
- `FIRInterpByN` / `StreamingFIRInterpByN` zero-stuffed and filtered L−1 low-rate states (~50 % of the
  per-hop time) → polyphase with min-phase taps, state ceil(L/F)−1. Realtime TFLite invoke 0.51 → 0.23
  ms/hop on the Mac; states bundle 28 047 → 27 767 floats. Old long-state graphs still load, output
  identical (1.5e-8).

Pedal (C++):
- `std::sinf/expf` broke GCC 12 builds — the round-04 version **never compiled on Pi OS**.
- Masked ring indices made full == empty → monotonic counters. A per-ring flush raced the dry-then-out
  push (DI a hop ahead after bypass) → `dropAligned`.
- The callback thread ran below inference (inherited FIFO 20) → callback 40, inference 35, main not RT.
- Per-hop `std::vector` allocations → precomputed copy plan; XNNPACK delegate freed after the
  interpreter; model loaded, validated and warmed in `main` before the stream opens.
- `-ffast-math` compiled away every `isfinite` → `-fno-finite-math-only`; NaN/clamp guard on the output.
- Jitter buffer + `--latency-hops N`; invalid device IDs no longer segfault; `--webui`, SIGTERM,
  pickup and FX loop in the web UI.

## Why / how it was verified
- Scratch harness: training vs streaming equivalence (hop 32 and 1024, ~4e-6), TFLite parity (~9e-6),
  a tiny end-to-end run (train offline → realtime KD → export/verify → magnetic fine-tune → `--process`
  → TFLite demos → stems → resume).
- Docker `debian:bookworm` arm64 with GCC 12 and TFLite 2.20 built via CMake: the C++ TFLite wrapper is
  bit-identical to Python; `MasterEqIIR` vs `master_eq_sos` ≤ 1.4e-5; a paced callback + inference test
  never misaligned DI and processed audio.
- The user rejected running the review with parallel reviewer subagents ("Don't do it with subagents"):
  the review was done by reading the files directly.

## What we learned
Deliberately **not** changed (reviewed, judged fine or not worth it):
- The style classifier trains on full segments; the deploy side smooths logits with an EMA.
- Auto style at hop 1024 (dynamic model in `--process` / `--demo`) differs slightly from hop 32; only a
  forced style is exact (documented).
- Baseline boosts stay synced constants; the pedal can't auto-detect fine-tuned models (8 vs 32 dB
  pick-reso baseline).
- Pedal output gain −3 dB vs Python 0 dB; upsampler taps sum to 1 (−18 dB, absorbed by learned gains);
  `StreamingStyleLerp`'s first sample repeats the previous hop (negligible).

## State at the end
All fixes in, verified off-target, uncommitted until round 07. The models in `out/models/` (2025-10-19) predate all of
it — they load and run, but none of the fixes are in their weights and their TFLite exports lack the
faster upsamplers. The pedal has never run on the real Pi hardware.

## Next
Retrain everything; first run on the Pi (`Proc=` vs the 0.67 ms budget, xruns; try `--latency-hops 2`).
The user's long-standing complaint about picked playing became round 06.
