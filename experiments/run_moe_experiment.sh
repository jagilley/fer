#!/bin/bash
# Run MoE CPPN experiment
# Compares MoE with 6 tiny experts (~5520 params) against dense small model (~5544 params)

cd "$(dirname "$0")/../src"

echo "=== Running MoE Conditional CPPN Experiment ==="
echo "Architecture: 6 tiny experts (6;cache:8,gaussian:2,identity:1,sin:1)"
echo "Total params: ~5520 (vs Small model: 5544)"
echo ""

python train_moe.py \
  --expert_arch "6;cache:8,gaussian:2,identity:1,sin:1" \
  --n_experts 6 \
  --router_hidden 48 \
  --n_images 3 \
  --img_files ../data/picbreeder_skull/img.png ../data/picbreeder_butterfly/img.png ../data/picbreeder_apple/img.png \
  --n_iters 100000 \
  --lr 3e-3 \
  --track_metrics \
  --metric_interval 10000 \
  --save_dir ../experiments/conditional_moe_6experts \
  --seed 0

echo ""
echo "=== Experiment Complete ==="
echo "Results saved to: experiments/conditional_moe_6experts/"
echo ""
echo "Compare against: experiments/conditional_small_direct/"
