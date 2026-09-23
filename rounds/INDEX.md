# Rounds

One line per round, newest last. Read the latest round in full (its **Next** section is the hand-off);
read an older round when its line touches what you're working on. How rounds work: `AGENTS.md`.

Rounds 01–06 were written retroactively in round 07 from git history, file dates, the old `TRAINING.md`
state sections and the previous agent's notes.

| Round | Date | Summary |
|---|---|---|
| [01](round_01.md) | 2025-06-10 → 2025-10-19 | Initial development: piezo→mic offline teacher + realtime KD student; the 1 h pick/finger takes; the models still in `out/models/` |
| [02](round_02.md) | 2025-11-25 → 2026-03-22 | Pedal on the Raspberry Pi + HiFiBerry (C++/TFLite, FX loop), `--process`, PySide6 desktop app |
| [03](round_03.md) | 2026-07-05 | First review pass; pickup-type conditioning (piezo/magnetic adapter + FiLM), data layout `input/<pickup>/<style>` |
| [04](round_04.md) | 2026-09-02 | Second pass: real non-causal KD teacher, contexted-core losses, min-phase FIRs, master EQ moved out of the models |
| [05](round_05.md) | 2026-09-22 | Third pass: KD alignment and `val_loss` fixes, polyphase upsamplers (2.2x faster), pedal builds on the Pi toolchain, ring/DI race fixes |
| [06](round_06.md) | 2026-09-23 | Pick attack: the mic hears the pick contact ~6.5 ms before the piezo; split phase/magnitude + adversarial offline loss, `--analyze-attacks`, pilot; retrain crashed the Mac |
| [07](round_07.md) | 2026-09-23 | Memory-safe training + `train_guard.py`; retrain postponed until magnetic triplets; electric-guitar strategy; `eq_match.py`; this rounds structure; future desktop app, DAW plugins (VST3/AU + ARA) and end-user fine-tuning in the backlog |
| [08](round_08.md) | 2026-09-23 | Stereo two-mic target (soundhole + 12th fret) decided: B-22; the next corpus is a four-track session (`TRAINING.md` §4); booth guide `RECORDING.md` |
