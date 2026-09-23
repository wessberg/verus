# AGENTS.md

How to work in this repo — for any agent or person. `CLAUDE.md` only imports this file.

## The project

A guitar pedal that turns a pickup signal into the sound of a mic'd acoustic guitar: an acoustic's
under-saddle piezo or magnetic soundhole pickup, and — the goal of the magnetic work — an electric
guitar's passive magnetic pickup. Two neural models in one file, `piezo2mic.py`: an offline teacher
(96 kHz, non-causal) and a realtime student (48 kHz, causal, 32-sample hops) distilled from it, which
runs on a Raspberry Pi + HiFiBerry DAC8x/ADC8x through TFLite in `cpp/src/main.cpp`. Planned
beyond the pedal: a standalone desktop app and DAW plugins that can also fine-tune models for a
user's own guitar (`BACKLOG.md`, "Future").

`TRAINING.md` is the runbook: how the system works and how to run every step. Read the sections for
the area you touch; §8 lists the sharp edges.

## Work happens in rounds

A **round** is one working session with a goal, ending in a hand-off. The next round may be a
different agent with none of your memory, often months later — everything it needs must be in the
repo, not in chat history or an agent's private memory.

| File | Holds |
|---|---|
| `rounds/INDEX.md` | one line per round: number, date, one-sentence summary |
| `rounds/round_NN.md` | the full record of round NN: what happened, why, what we learned, what's next |
| `BACKLOG.md` | every open task, prioritized, with stable ids (`B-7`) |
| `TRAINING.md` | the runbook — how things work *now*; not a history, not a to-do list |

### Starting a round
1. Read `rounds/INDEX.md`, then the latest round file in full — its **Next** section is the hand-off.
   Read older rounds when their index line touches your task.
2. Read `BACKLOG.md`.
3. `git status` and `git log -5`: a round may have ended with uncommitted work (the round file says).
4. Create `rounds/round_NN.md` (next number, two digits) with the header and goal, and add its line to
   `rounds/INDEX.md` right away, so an interrupted round still leaves a trace.

### During a round
Keep the round file current as you go (append to **What happened** at each milestone), so a crash or
a context reset loses nothing. Record measurements with their numbers and how they were obtained.

### Ending a round — or handing off mid-way
1. Complete `rounds/round_NN.md`; the **Next** section must let a cold reader continue.
2. Update the round's line in `rounds/INDEX.md` (one sentence).
3. Update `BACKLOG.md`: delete finished items (the round file records them), add new ones with the
   next free id, re-prioritize.
4. Update `TRAINING.md` if how anything works or is run has changed.
5. State in the round file what is committed and what isn't.

Round files are records: don't rewrite old ones. If an old round turns out wrong, say so in the
current round (and add a one-line "Correction: see round NN" to the old one).

### Round file template

```markdown
# Round NN — <title>

- **Date:** YYYY-MM-DD
- **Agent:** <model / person>
- **Goal:** what this round set out to do, and why
- **Commits:** <hashes, or "none — uncommitted: <files>">

## What happened
Chronological, including dead ends and what stopped them.

## Why
The decisions and the evidence or reasoning behind them.

## What we learned
Findings that outlive the round: measurements (with numbers), root causes, what didn't work.

## State at the end
What exists now — models, data, tools, what is verified, what is running.

## Next
The hand-off: what to do first, with BACKLOG ids.
```

## Rules for this repo

### The dev machine — read before running anything heavy
Apple Silicon MacBook, **16 GB unified memory**, ~460 GB SSD that is usually **96–99 % full**. In
round 06 a training run swapped until the SSD filled and macOS crashed.
- Before a heavy job: `df -h /System/Volumes/Data`, `sysctl vm.swapusage`, available memory. With the
  user's apps open only ~6–10 GB is available.
- Probe a new workload's memory with a short guarded run first; run long jobs under `train_guard.py`
  (usage: `TRAINING.md` §5 Step 1). Never two heavy TensorFlow jobs at once.
- Judge memory by system available memory and swap, not process RSS: Metal (GPU) allocations come
  from the same unified memory and don't show in RSS. Train with `--cpu`; offline batch 1 (round 07).
- Stream audio in blocks; never hold the corpus (4 × 1 h at 96 kHz) in RAM. Keep scratch data small
  and delete it. Never delete the user's files to make room without asking.

### Code
- `piezo2mic.py` is one file on purpose. Match its style: comments that explain *why*, constants at the
  top of their section, standalone tools (`train_guard.py`, `eq_match.py`) with a usage docstring.
- Python ↔ C++ constants must stay in sync (`TRAINING.md` §7, "Constants that must stay in sync").
- After any change to the realtime streaming state: `--inspect-tflite-all` must print
  "OK: bundle matches streaming state sizes" (contract: `TRAINING.md` §8).
- Verify with a scratch harness (small or synthetic data, equivalence checks) before any long run, and
  write in the round file what was verified and how.
- The pedal can be compile-checked off the Pi: Docker `debian:bookworm` arm64 with `g++` and
  `portaudio19-dev`, TensorFlow v2.20.0 source + flatbuffers v24.3.25 headers (TFLite's CMake needs
  `TF_MAJOR_VERSION=2 TF_MINOR_VERSION=20 TF_PATCH_VERSION=0 TF_VERSION_SUFFIX=""`). Docker Desktop is
  installed on the dev Mac but normally stopped.

### Working with the user
- Months can pass between rounds: assume a cold start and lead with where things were left.
- The user is at home in DSP and ML detail and prefers thorough, autonomous execution over
  check-ins. Ask only for what is genuinely theirs: recordings, listening, hardware, direction.
- For recordings and listening tests, say exactly what to record or listen for, and why.
- Reviews and multi-file fixes: read the code yourself, in one context — the user turned down
  parallel reviewer subagents (round 05).
- Commit, push and merge only when the user asks. `master` is the main branch.

## Where things are

| Path | |
|---|---|
| `piezo2mic.py` | models, training, export, `--process`, analysis commands (`TRAINING.md` §2) |
| `cpp/` | the pedal runtime; build: `cpp/readme.md` |
| `piezo2mic_app/` | PySide6 desktop converter (shells out to `--process`) |
| `train_guard.py` | memory / swap / disk watchdog for long jobs |
| `eq_match.py` | acoustic magnetic → electric humbucker EQ match ("fake electric" sources) |
| `input/` | training audio (layout: `TRAINING.md` §4); `input/eqmatch/` is not read by training |
| `out/models/` | trained models; `out/models/offline/legacy_2025-10/` = backup of the Oct-2025 offline model |
| `out/checkpoints/`, `out/logs/`, `out/analysis/` | checkpoints, training logs, `--analyze-attacks` results |
| `readme.md` | model I/O and the HiFiBerry setup on the Pi |
