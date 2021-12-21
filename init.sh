#!/bin/sh
pip install -r requirements.txt
pip install mesh-transformer-jax/ jax==0.2.12
pip3 install tf-nightly --no-input
gsutil cp -r gs://ks-story-ew4-storage/step_383500 .
python3 gen.py