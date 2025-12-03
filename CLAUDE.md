# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements experiments on Fractured Entangled Representations (FER) vs Unified Factored Representations (UFR) in neural networks. It compares CPPNs (Compositional Pattern-Producing Networks) trained via SGD against those evolved through open-ended search (Picbreeder), analyzing how internal representations differ despite producing identical outputs.

Key research question: Does model scaling affect representation quality? Can SGD learn more UFR-like representations?

## Common Commands

### Environment Setup
```bash
conda create --name=fer python=3.10.16 --yes
conda activate fer

# JAX with CUDA (for GPU training)
python -m pip install jax==0.4.28 jaxlib==0.4.28+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --no-cache-dir
python -m pip install flax==0.10.2 evosax==0.1.6 orbax-checkpoint==0.11.0 optax==0.2.4 --no-deps
python -m pip install -r requirements.txt
```

### Training

Train single-image CPPN:
```bash
cd src
python train_sgd.py --arch="12;cache:15,gaussian:4,identity:2,sin:1" --img_file=../data/picbreeder_skull/img.png --save_dir=../data/my_run
```

Train conditional CPPN (multiple images in one network):
```bash
cd src
python train_conditional.py \
  --arch="6;cache:8,gaussian:2,identity:1,sin:1" \
  --n_images=3 \
  --img_files ../data/picbreeder_skull/img.png ../data/picbreeder_butterfly/img.png ../data/picbreeder_apple/img.png \
  --save_dir=../data/my_conditional_run \
  --track_metrics
```

### Running Tests
```bash
cd src
python test_conditional.py
```

## Architecture

### CPPN Architecture String Format
Architecture is specified as `"<layers>;<activation>:<count>,..."`:
- Example: `"12;cache:15,gaussian:4,identity:2,sin:1"` = 12 layers, each with 15 cache + 4 gaussian + 2 identity + 1 sin neurons
- Available activations: `cache` (identity), `identity`, `cos`, `sin`, `tanh`, `sigmoid`, `gaussian`, `relu`

### Core Modules

**`src/cppn.py`**: Standard CPPN implementation
- `CPPN`: Flax module generating single images from (x, y, d, b) coordinates
- `FlattenCPPNParameters`: Wrapper using evosax to flatten params for analysis

**`src/cppn_conditional.py`**: Conditional CPPN for multi-image generation
- `ConditionalCPPN`: Extends CPPN with one-hot image_id conditioning
- Input: base coords + one-hot vector for image selection

**`src/train_sgd.py`**: Training script for single-image CPPNs
- Uses Adam optimizer with normalized gradients
- Outputs: params.pkl, losses.pkl, img.png

**`src/train_conditional.py`**: Training script for conditional CPPNs
- Two modes: `direct` (train from scratch) or `distill` (from teacher)
- Optional FER/UFR metric tracking during training

**`src/fer_metrics.py`**: FER/UFR representation metrics
- `feature_similarity`: Correlation between feature maps (high = UFR) - conditional only
- `neuron_specialization`: Coefficient of variation (high = FER) - conditional only
- `interpolation_smoothness`: Smoothness along conditioning simplex (high = UFR) - conditional only
- `spatial_roughness`: **KEY METRIC** - measures spatial smoothness of feature maps (low = UFR)

**`src/train_distill.py`**: Distillation training (teacher → student)
- Phase 1: Train student on teacher outputs at simplex-sampled points
- Phase 2: Fine-tune on ground truth images

**`src/post_training_viz.py`**: Auto-generates visualizations after training
- Edge interpolations, simplex sampling, feature maps

### Data Flow
1. Images loaded as (H, W, 3) RGB arrays via matplotlib
2. CPPN generates from coordinate grid via `generate_image(params, img_size=256)`
3. HSV output converted to RGB via `color.hsv2rgb`
4. Results saved as .pkl (params, losses) and .png (images)

### Key Directories
- `src/`: Core Python code
- `data/`: Training outputs and Picbreeder layerized CPPNs
- `experiments/`: Training scripts with specific configurations
- `picbreeder_genomes/`: Raw Picbreeder genome files
- `assets/`: Paper figures and visualizations

## Key Research Findings

### Spatial Roughness as UFR/FER Metric

**The most important discovery**: Spatial roughness of feature maps is the best quantitative indicator of UFR vs FER.

- **Roughness** = average gradient magnitude across feature maps (how much adjacent pixels differ)
- **UFR (Picbreeder)**: Very smooth feature maps, roughness ~0.005-0.01, max roughness <0.10
- **FER (SGD-trained)**: Noisy/patchy feature maps, roughness ~0.04-0.06, max roughness >0.15

Why this matters:
1. **Only metric that works for both single-image and conditional CPPNs** - other metrics require multiple outputs
2. **Directly measures "clean functional composition"** - UFR networks compute smooth transformations, FER networks develop noisy specialized circuits
3. **Matches qualitative claims in original FER paper** - they described "random bullshit" in intermediate layers but didn't quantify it

### Model Size Findings

- **Tiny models** (840 params) have lower spatial roughness than larger models - capacity constraints force smoother representations
- **Distillation** from tiny → small partially transfers smoothness, but effect is limited
- **All SGD-trained models** are far from Picbreeder-level smoothness (5-10x higher roughness)

### What Doesn't Work as UFR Metrics

- **Max activation magnitude**: Differs between Picbreeder and SGD, but likely just a process artifact (NEAT vs gradient descent), not indicative of representation quality
- **Feature similarity/neuron specialization**: Only work for conditional CPPNs, don't generalize
