# Python dependencies

`pip install soundfile tflite`

# Dead code removal

`vulture piezo2mic.py`

# Monitor training

tensorboard --logdir=./out/logs --port=6006

# Timeout prevention for ROCm

./rocm-idle-protection.sh