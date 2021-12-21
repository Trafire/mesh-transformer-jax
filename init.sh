#!/bin/sh
pip install mesh-transformer-jax/ jax==0.2.12 tensorflow==2.5.0
pip install -r requirements.txt
gsutil cp -r gs://ks-story-ew4-storage/step_383500 .
pip3 uninstall keras
pip3 install keras --upgrade
pip3 install tf-nightly
python3 gen.py