#!/bin/sh
pip3 install -r requirements.txt
pip3 install ../mesh-transformer-jax/
pip3 install jax==0.2.12 tensorflow==2.5.0
pip3 install tf-nightly --no-input
gsutil cp -r gs://ks-story-ew4-storage/step_383500 .
python3 gen.py