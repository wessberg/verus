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

`pip install soundfile tflite`

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

