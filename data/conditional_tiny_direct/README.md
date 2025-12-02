# Conditional CPPN - Tiny Direct Training

## Experiment Overview

This experiment trains a tiny conditional CPPN (840 parameters) directly on three Picbreeder images to establish a baseline for FER (Fractured Entangled Representation) characteristics under extreme capacity pressure.

## Model Architecture

- **Architecture**: `6;cache:8,gaussian:2,identity:1,sin:1`
- **Parameters**: 840 total (15% of skull-sized baseline)
- **Conditioning**: One-hot encoding at input layer (early fusion)
- **Images**: 3 (skull, butterfly, apple)

## Training Configuration

```bash
python train_conditional.py \
  --seed=0 \
  --save_dir="../data/conditional_tiny_direct" \
  --arch="6;cache:8,gaussian:2,identity:1,sin:1" \
  --n_images=3 \
  --img_files ../data/picbreeder_skull/img.png ../data/picbreeder_butterfly/img.png ../data/picbreeder_apple/img.png \
  --n_iters=100000 \
  --lr=0.003 \
  --track_metrics \
  --metric_interval=5000
```

## Key Parameters

- **Training iterations**: 100,000
- **Learning rate**: 0.003 (Adam)
- **Metric tracking**: Every 5,000 steps
- **Gradient normalization**: Enabled
- **Mode**: Direct (training from scratch, not distillation)

## Expected Outcomes

- **Strong FER characteristics**: Capacity pressure should force task-specific circuits
- **Low feature similarity**: Different images should use different internal representations
- **High neuron specialization**: Neurons should activate selectively for specific images
- **Low interpolation smoothness**: Blends between training images should be chaotic

## Results Summary

- **Final Loss**: ~0.003
- **Feature Similarity**: 0.4175 (FER, but higher than 5.5K model's 0.2867)
- **Neuron Specialization**: 0.3113 (moderate FER)
- **Interpolation Smoothness**: 0.3746 (low, confirming FER)

## Training Dynamics

- Feature similarity starts at 0.52 and decreases to 0.42 (fracturing during training)
- Neuron specialization remains stable around 0.31
- Interpolation smoothness degrades from 0.46 to 0.37

## Purpose

This baseline will be compared against distillation experiments where a large teacher model provides smooth interpolation targets. The hypothesis is that distillation should produce more UFR-like representations even in the same tiny architecture.
