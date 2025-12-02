#!/bin/bash
# Training script for conditional CPPN - Medium model
# Architecture: 9;cache:11,gaussian:3,identity:2,sin:1 (between tiny and small)

cd ../../src

python train_conditional.py \
  --seed=0 \
  --save_dir="../data/conditional_medium_direct" \
  --arch="9;cache:11,gaussian:3,identity:2,sin:1" \
  --n_images=3 \
  --img_files ../data/picbreeder_skull/img.png ../data/picbreeder_butterfly/img.png ../data/picbreeder_apple/img.png \
  --n_iters=100000 \
  --lr=0.003 \
  --track_metrics \
  --metric_interval=5000 \
  --mode=direct
