# Contributing

Start with [TRAINING.md](docs/TRAINING.md) (how the system works and how to run each step; §8 lists the
sharp edges) and [ROADMAP.md](docs/ROADMAP.md).

## Code conventions

- The Python package is `src/verus/` (module map: [docs/TRAINING.md](docs/TRAINING.md) §1). Constants go
  in `config.py`. Match the existing style: comments that explain *why*; standalone tools in `tools/`
  (`train_guard.py`, `eq_match.py`) with a usage docstring.
- Install in editable mode for development: `pip install -e .`
- Python ↔ C++ constants must stay in sync ([TRAINING.md](docs/TRAINING.md) §7, "Constants that must stay
  in sync").
- After any change to the realtime streaming state, `verus --inspect-tflite-all` must
  print `OK: bundle matches streaming state sizes` (contract: [TRAINING.md](docs/TRAINING.md) §8).
- Verify with a small or synthetic harness (equivalence checks) before any long training run.
- The pedal can be compile-checked off the Pi: Docker `debian:bookworm` arm64 with `g++` and
  `portaudio19-dev`, TensorFlow v2.20.0 source and flatbuffers v24.3.25 headers (TFLite's CMake needs
  `TF_MAJOR_VERSION=2 TF_MINOR_VERSION=20 TF_PATCH_VERSION=0 TF_VERSION_SUFFIX=""`). Build
  instructions: [runtime/README.md](runtime/README.md).

## Running heavy jobs

Training is memory- and disk-hungry. On a 16 GB machine:

- Check free disk, swap and available memory first; run long jobs under `tools/train_guard.py` (usage:
  [TRAINING.md](docs/TRAINING.md) §5 Step 1) and never run two heavy TensorFlow jobs at once.
- Train with `--cpu` and offline batch size 1. Judge memory by system available memory and swap, not
  process RSS: Metal (GPU) allocations come from the same unified memory and don't show in RSS.
- Stream audio in blocks; don't hold the corpus in RAM.
