# Start here

**See [TRAINING.md](TRAINING.md)** for the full training / inference / Raspberry-Pi
deployment runbook (data layout, command order, sharp edges).

Where things stand and what's next: [rounds/INDEX.md](rounds/INDEX.md) (one line per working
round; the latest round's file ends with the hand-off) and [BACKLOG.md](BACKLOG.md) (open work).
[AGENTS.md](AGENTS.md) describes how work is organized, for agents and people alike.

# Model outputs

The realtime TFLite model has 4 outputs (there is exactly ONE 3-D output, so runtimes can pick
the audio tensor by shape when TFLite drops the names):

1. Processed mono audio
   - Shape: [1, T, 1]
   - Name: realtime_audio_out

2. Style probability (pick vs finger, EMA-smoothed by the model)
   - Shape: [1, 1]
   - Name: realtime_style_prob_out

3. Packed streaming state bundle (lexicographic order of the `realtime_state_*` inputs)
   - Shape: [1, N] (2‑D float)
   - Name: realtime_states_bundle_out

4. Splits (lengths of each state in the bundle)
   - Shape: [1, Nstates] or [Nstates] (int32)
   - Name: realtime_states_splits_out

The offline TFLite model has 2 outputs: audio [1, T, 1] and style probability [1, 1].
(The former "dry" output was a copy of the input — the models are zero-latency — and was removed
because a second [1, T, 1] tensor made the audio output ambiguous.)

The master EQ is not part of the models: the pedal applies it post-model (`MasterEqIIR` in
`cpp/src/main.cpp`) and `--process --eq ...` applies the identical biquads in Python.

# Python dependencies

`pip install -r requirements.txt`

(TensorFlow with Keras 3, ai-edge-litert, numpy, scipy, soundfile; add pyside6 for the UI.)

# Dead code removal

`vulture piezo2mic.py`

# Pico interaction from Raspberry

You can run a script on it:

```
mpremote connect /dev/serial/by-id/usb-MicroPython_Board_in_FS_mode_3ec34210277d08ef-if00 run ./demo_color_palette.py
```

# Check hifibery stuff:

1. Create this file: `/etc/asound.conf`:

```
pcm.!default {
  type hw card 0
}
ctl.!default {
  type hw card 0
}
```

2. In /boot/firmware/config.txt:

- Comment out: `dtparam=audio=on`
- Change `dtoverlay=vc4-kms-v3d` to `dtoverlay=vc4-kms-v3d,noaudio`
- Add:
  ```
  # Hifiberry
  dtoverlay=hifiberry-dac8x
  force_eeprom_read=0
  ```
- Add this in the end if missing:
  ```
  [all]
  usb_max_current_enable=1
  ```

3. Reboot

4. Run `aplay -l`:

```
aplay -l
**** List of PLAYBACK Hardware Devices ****
card 0: sndrpihifiberry [snd_rpi_hifiberry_dac8x], device 0: HiFiBerry DAC8xADC8x HiFi snd-soc-dummy-dai-0 [HiFiBerry DAC8xADC8x HiFi snd-soc-dummy-dai-0]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
```

5. Also run `arecord -l`:

```
arecord -l
**** List of CAPTURE Hardware Devices ****
card 0: sndrpihifiberry [snd_rpi_hifiberry_dac8x], device 0: HiFiBerry DAC8xADC8x HiFi snd-soc-dummy-dai-0 [HiFiBerry DAC8xADC8x HiFi snd-soc-dummy-dai-0]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
```

# Processing audio through a model to disk as a new file

### CLI

Auto style:
```
python piezo2mic.py --process --input in.wav --output out.wav --offline-only
```

Force pick:
```
python piezo2mic.py --process --input in.wav --output out.wav --offline-only --style pick
```

Force finger
```
python piezo2mic.py --process --input in.wav --output out.wav --offline-only --style finger
```

Magnetic pickup source (e.g. an electric guitar; `electric` is an alias for `magnetic`):
```
python piezo2mic.py --process --input in.wav --output out.wav --offline-only --pickup magnetic
```

With the pedal's master EQ (low cut 80 Hz @ 24 dB/oct, +2 dB high shelf):
```
python piezo2mic.py --process --input in.wav --output out.wav --offline-only --eq lc_hz=80,lc_slope=24,hs_gain_db=2
```

# Training data layout

```
input/<pickup>/<style>/{source,target}/takeN_source.wav + takeN_target.wav
```

with `pickup` in `piezo|magnetic` and `style` in `pick|finger`. The legacy layout
`input/<style>/{source,target}` is still read and treated as piezo. Record piezo + magnetic +
mic simultaneously where possible so both sources share the identical mic target.

### UI

You can also du this via a UI:

```
pip install pyside6
python -m piezo2mic_app
```