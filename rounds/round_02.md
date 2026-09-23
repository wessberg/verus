# Round 02 — Pedal on the Raspberry Pi, file processing, desktop app

- **Date:** 2025-11-25 → 2026-03-22
- **Agent:** the user
- **Goal:** run the realtime model as a pedal on a Raspberry Pi with a HiFiBerry DAC8x/ADC8x.
- **Commits:** `c4d1510e` … `fa61b52a`

*Written retroactively in round 07 from git history.*

## What happened
- 2025-11-25: "pi-oriented changes": `cpp2/` → `cpp/` with a CMake build for the Pi; an FX loop in the
  pedal.
- 2025-11-29: DAC output 8 carries a gain-normalized mono sum of the main L/R outputs.
- 2026-01: Pi setup documented (`readme.md` HiFiBerry config, `cpp/readme.md` build).
- 2026-03-22: `--process` (file processing CLI in `piezo2mic.py`) and the PySide6 desktop converter
  `piezo2mic_app/`, which shells out to it.

## State at the end
Pedal code for the Pi and desktop tooling exist; the models are still the 2025-10-19 ones.

## Next
A full review of the code (round 03).
