# Round 06 — Pick attack: why picked playing fails; a new offline objective

- **Date:** 2026-09-23
- **Agent:** Claude (Claude Code)
- **Goal:** the user's long-standing complaint: fingerstyle sounds great (especially offline), picked
  playing never did. Find out why and fix the offline model first.
- **Commits:** committed in round 07 together with rounds 05 and 07 ("Rounds 05-07: review fixes, pick-attack objective, memory-safe training, EQ match, rounds docs"), merged into `master`.

*Written retroactively in round 07 from the previous agent's notes and the old `TRAINING.md` §9b/§10.*

## What happened
1. Measured the pick/finger takes (1 h each, 96 kHz, aligned to 0.00 samples) and the legacy models
   with a new command, `--analyze-attacks` (3 held-out excerpts per group, JSON in `out/analysis/`,
   diffed against the previous run). Baselines for the legacy models are in `out/analysis/`.
2. Split the offline objective at `OFFLINE_PHASE_LOSS_CUTOFF_HZ` = 3000: phase-sensitive terms (L1,
   HF, D2, complex STFT) only below; a short-window log-magnitude MR-STFT (`lmag`) above.
3. `OFFLINE_ADVERSARIAL`: three-resolution log-spectrogram LSGAN discriminators with feature matching
   (`AdversarialTrainer`, `AdversarialRamp`). Adversarial runs keep the LAST weights
   (`EarlyStopping(restore_best_weights=False)`), checkpoint every epoch (`GeneratorCheckpoint`), and
   `--promote-offline-checkpoint` exports a chosen checkpoint.
4. A pilot (below), then a full offline retrain was started — batch 4, corpus loaded in RAM. It swapped
   the Mac until the SSD filled and **macOS crashed**. Round 07 picked up from there.

## Why
A deterministic, phase-sensitive regression can predict almost nothing of HF content that isn't
predictable from the input; it learns to output the average (quiet, dull attacks). Supervising HF by
magnitude and realism instead of waveform lets the model produce plausible HF without matching phase.

## What we learned
- **The mic's attack HF leads the piezo's by a median 6.4–6.9 ms** on picked playing (5.8–6.3 ms
  fingerstyle), consistently along the takes: the mic hears the pick/nail *contact*, the saddle piezo
  only the string *release*; no LF cue in the piezo in between. The pick's contact click is loud
  (attack windows ~17 dB above the rest in 6–12 kHz, vs ~8 dB fingerstyle).
- Piezo↔mic phase coherence at pick attacks above 6 kHz: 0.03–0.19 depending on where in the take
  (fingerstyle 0.12–0.43) — the fine structure is largely unpredictable; the energy envelope is not
  (per-attack HF energy correlation ~0.94). An early single-window figure of "1–2 %" was
  unrepresentative and was corrected with the user.
- The legacy offline model: attack HF 12–13 dB short of the mic at 0 dB knobs; its pick-attack branch
  keys on the release (4–30 ms late) and, with the +38 dB runtime boost, puts most of its energy
  *between* attacks.
- **Pilot** (legacy offline weights + 450 steps per arm, batch 1, 4-minute excerpts, scored
  `--as-trained` on the held-out tail at the last step):

  | objective | pick attack HF (6–20 kHz) | pick between attacks | pick 1–3 kHz | finger attack HF |
  |---|---|---|---|---|
  | old model (start) | −9.0 dB | −6.5 | −2.8 | −4.1 |
  | old loss, +450 steps | −9.0 | −6.5 | −3.2 | −4.8 |
  | split loss | −5.8 | −3.3 | −4.7 | −2.7 |
  | split + adversarial | **−4.2** | −1.9 | −2.7 | **−1.2** |

  The new objective moves attack HF toward the mic (+4.8 dB pick, +3.6 dB finger vs the old loss at the
  same budget); phase coherence doesn't change (it isn't asked to). The HF time course still peaks on
  the release, not the contact (BACKLOG B-11). Reconstruction `val_loss` rose ~10 % with the
  discriminator — expected, so adversarial epochs are selected by `--analyze-attacks` and by ear.
  With the runtime pick boosts (+38 dB) the new model is ~5 dB over-bright between attacks → re-tune the
  baseline after retraining (B-10). A pilot, not a verdict: short, from old weights, not listened to.

## State at the end
New objective implemented (offline only), pilot done, no retrained model. The retrain attempt crashed
the machine; nothing from it survived.

## Next
Retrain safely (round 07). Then the pick-click generator branch, the realtime timing decision, and the
magnetic pickup as a contact cue (B-11, B-12, B-13).
