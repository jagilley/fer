# Conditional CPPN - Small (Skull-Sized) Direct Training

## Experiment Overview

This experiment trains a small conditional CPPN (5,544 parameters) directly on three Picbreeder images. This uses the "skull-sized" architecture that successfully compressed individual images in prior experiments, now extended to handle multiple images via conditioning.

## Model Architecture

- **Architecture**: `12;cache:15,gaussian:4,identity:2,sin:1` (skull-sized)
- **Parameters**: 5,544 total (5,478 base + 66 for conditioning)
- **Conditioning**: One-hot encoding at input layer (early fusion)
- **Images**: 3 (skull, butterfly, apple)

## Training Configuration

```bash
python train_conditional.py \
  --seed=0 \
  --save_dir="../data/conditional_small_direct" \
  --arch="12;cache:15,gaussian:4,identity:2,sin:1" \
  --n_images=3 \
  --img_files ../data/picbreeder_skull/img.png ../data/picbreeder_butterfly/img.png ../data/picbreeder_apple/img.png \
  --n_iters=100000 \
  --lr=0.003 \
  --mode=direct
```

## Key Parameters

- **Training iterations**: 100,000
- **Learning rate**: 0.003 (Adam)
- **Metric tracking**: Not enabled during training (computed post-hoc)
- **Gradient normalization**: Enabled
- **Mode**: Direct (training from scratch, not distillation)

## Expected Outcomes

- **Strong FER characteristics**: Capacity pressure should force task-specific circuits
- **Low feature similarity**: Different images should use different internal representations
- **High neuron specialization**: Neurons should activate selectively for specific images
- **Low interpolation smoothness**: Blends between training images should be chaotic

## Results Summary

- **Final Loss**: 0.00295
- **Feature Similarity**: 0.2867 (strong FER, below 0.3 threshold)
- **Neuron Specialization**: ~0.31 (moderate FER)
- **Interpolation Smoothness**: 0.4159 (low, confirming FER)

## Key Findings

This baseline demonstrates **clear FER characteristics** when training under capacity pressure:

1. **Feature similarity is low** (0.29): Different images use largely independent internal representations
2. **Interpolations are chaotic**: Blending between training points produces incoherent results
3. **Neuron specialization is moderate**: Neurons activate selectively for specific images

The model successfully fits all three images but does so by learning entangled, task-specific circuits rather than unified factored representations.

## Comparison Notes

This model serves as the baseline for comparison with:
- **Tiny model (840 params)**: Tests extreme capacity constraints
- **Distillation experiments**: Tests whether teacher-provided interpolation targets can induce UFR

Surprisingly, the tiny model (15% the size) showed higher feature similarity (0.42) than this model (0.29), though both exhibit clear FER characteristics.

## Purpose

This experiment establishes the FER baseline for direct training under moderate capacity pressure. The skull-sized architecture was chosen because it successfully compressed individual images in prior experiments, making it a natural starting point for multi-image conditioning.
