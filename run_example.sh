#!/usr/bin/env bash

export PYTORCH_ENABLE_MPS_FALLBACK=1
python3 -m venv venv
source venv/bin/activate
pip install --require-virtualenv -r requirements.txt
pip install --require-virtualenv deepfloyd_if==1.0.2rc0 --no-deps
pip install --require-virtualenv git+https://github.com/openai/CLIP.git --no-deps
python example.py