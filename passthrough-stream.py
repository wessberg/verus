import sounddevice as sd
import numpy as np
import time

SR       = 48000
BLOCK    = 128
CHANNELS = 1
DEVICE   = 3      # set your device

def callback(indata, outdata, frames, t0, status):
    if status:
        print("⚠️", status)
    # simple pass-through
    outdata[:] = indata

with sd.Stream(samplerate=SR,
               blocksize=BLOCK,
               device=DEVICE,
               dtype='float32',
               channels=CHANNELS,
               latency='low',
               callback=callback):
    print("Pass-through running… press Ctrl-C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass