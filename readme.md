# Start here

**See [TRAINING.md](TRAINING.md)** for the full training / inference / Raspberry-Pi
deployment runbook (data layout, command order, sharp edges, magnetic-pickup roadmap).

# Model outputs

The model has 5 outputs:

1. Processed mono audio
   - Shape: [1, T, 1]
   - Name: realtime_audio_out

2. Style probability (pick vs finger)
   - Shape: [1, 1]
   - Name: realtime_style_prob_out

3. Unprocessed / dry mono signal, time‑aligned to model I/O
   - Shape: [1, T, 1]
   - Name: realtime_dry_out_export

4. Packed streaming state bundle
   - Shape: [1, N] (2‑D float)
   - Name: realtime_states_bundle_out

5. Splits (lengths of each state in the bundle)
   - Shape: [1, Nstates] or [Nstates] (int32)
   - Name: realtime_states_splits_out

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