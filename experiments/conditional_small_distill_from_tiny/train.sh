#!/bin/bash
# Distill small model from tiny teacher, then fine-tune on ground truth

cd ../../src

python train_distill.py \
    --seed 0 \
    --save_dir ../experiments/conditional_small_distill_from_tiny \
    --teacher_dir ../experiments/conditional_tiny_direct \
    --student_arch "12;cache:15,gaussian:4,identity:2,sin:1" \
    --img_files ../data/picbreeder_skull/img.png ../data/picbreeder_butterfly/img.png ../data/picbreeder_apple/img.png \
    --distill_iters 3000 \
    --n_simplex_samples 15 \
    --distill_lr 3e-3 \
    --finetune_iters 50000 \
    --finetune_lr 1e-3 \
    --track_metrics \
    --metric_interval 2000
