"""
Post-training visualization utilities.
Called automatically at the end of training to generate standard visualizations.
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import jax
import jax.numpy as jnp
from functools import partial

from color import hsv2rgb


def generate_at_condition(cppn, params, condition, img_size=128):
    """
    Generate image at arbitrary simplex conditioning.

    Args:
        cppn: FlattenConditionalCPPNParameters wrapper
        params: Flattened parameters
        condition: Array of shape (n_images,) with weights summing to 1
        img_size: Output resolution
    """
    # Unflatten params
    params_unflat = cppn.param_reshaper.reshape_single(params)

    # Generate coordinate grid
    x = y = jnp.linspace(-1, 1, img_size)
    grid_x, grid_y = jnp.meshgrid(x, y, indexing='ij')

    inputs = {}
    inputs['x'] = grid_x
    inputs['y'] = grid_y
    inputs['d'] = jnp.sqrt(grid_x**2 + grid_y**2) * 1.4
    inputs['b'] = jnp.ones_like(grid_x)

    # Add conditioning weights
    for i in range(len(condition)):
        inputs[f'img_{i}'] = jnp.full_like(grid_x, condition[i])

    # Construct input vector
    base_inputs = [inputs[name] for name in cppn.cppn.inputs.split(",")]
    cond_inputs = [inputs[f'img_{i}'] for i in range(len(condition))]
    all_inputs = base_inputs + cond_inputs
    inputs_stacked = jnp.stack(all_inputs, axis=-1)

    # Generate image
    (h, s, v), _ = jax.vmap(jax.vmap(partial(cppn.cppn.apply, params_unflat)))(inputs_stacked)
    r, g, b = hsv2rgb((h+1)%1, s.clip(0,1), jnp.abs(v).clip(0, 1))
    rgb = jnp.stack([r, g, b], axis=-1)

    return rgb


def generate_edge_interpolations(cppn, params, n_images=3, img_size=128):
    """Generate interpolations along simplex edges."""
    params_unflat = cppn.param_reshaper.reshape_single(params)

    fig, axes = plt.subplots(3, 7, figsize=(21, 9))

    names = ["Skull", "Butterfly", "Apple"][:n_images]
    edges = [
        (names[0], names[1], 0, 1),
        (names[0], names[2], 0, 2),
        (names[1], names[2], 1, 2)
    ]

    for edge_idx, (name1, name2, idx1, idx2) in enumerate(edges):
        alphas = np.linspace(0, 1, 7)

        for col_idx, alpha in enumerate(alphas):
            condition = np.zeros(n_images)
            condition[idx1] = 1 - alpha
            condition[idx2] = alpha

            img = generate_at_condition(cppn, params, condition, img_size)

            axes[edge_idx, col_idx].imshow(np.array(img))
            axes[edge_idx, col_idx].axis('off')

            if edge_idx == 0:
                if col_idx == 0:
                    axes[edge_idx, col_idx].set_title(f"{name1}\n[1,0,0]", fontsize=10)
                elif col_idx == 6:
                    axes[edge_idx, col_idx].set_title(f"{name2}\n[0,1,0]", fontsize=10)
                else:
                    axes[edge_idx, col_idx].set_title(f"α={alpha:.2f}", fontsize=9)

            if col_idx == 0:
                axes[edge_idx, col_idx].text(-0.1, 0.5, f"{name1}↔{name2}",
                                              transform=axes[edge_idx, col_idx].transAxes,
                                              fontsize=12, fontweight='bold',
                                              rotation=90, va='center', ha='right')

    plt.suptitle("Edge Interpolations", fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def generate_simplex_sampling(cppn, params, n_images=3, img_size=64):
    """Sample points throughout the simplex."""
    params_unflat = cppn.param_reshaper.reshape_single(params)

    n_levels = 5
    samples = []

    for i in range(n_levels):
        for j in range(n_levels - i):
            k = n_levels - 1 - i - j
            condition = np.array([i, j, k]) / (n_levels - 1)
            samples.append(condition)

    images = []
    for condition in samples:
        img = generate_at_condition(cppn, params, condition, img_size)
        images.append(np.array(img))

    # Create triangular layout
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.axis('off')

    img_idx = 0
    for i in range(n_levels):
        for j in range(n_levels - i):
            k = n_levels - 1 - i - j
            condition = np.array([i, j, k]) / (n_levels - 1)

            x = condition[2] + condition[0] * 0.5
            y = condition[0] * np.sqrt(3) / 2

            img = images[img_idx]
            img_idx += 1

            extent = [x - 0.08, x + 0.08, y - 0.08, y + 0.08]
            ax.imshow(img, extent=extent, zorder=10)

            from matplotlib.patches import Rectangle
            rect = Rectangle((x - 0.08, y - 0.08), 0.16, 0.16,
                           linewidth=1.5, edgecolor='black', facecolor='none', zorder=11)
            ax.add_patch(rect)

    # Draw simplex triangle
    triangle = plt.Polygon([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]],
                          fill=False, edgecolor='gray', linewidth=2, linestyle='--', zorder=1)
    ax.add_patch(triangle)

    # Label vertices
    names = ["Skull", "Butterfly", "Apple"]
    ax.text(0, 0, f"{names[1]}\n[0,1,0]", ha='center', va='top', fontsize=14, fontweight='bold')
    ax.text(1, 0, f"{names[2]}\n[0,0,1]", ha='center', va='top', fontsize=14, fontweight='bold')
    ax.text(0.5, np.sqrt(3)/2, f"{names[0]}\n[1,0,0]", ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.title("Simplex Sampling", fontsize=16, fontweight='bold', pad=20)
    return fig


def generate_feature_maps(cppn, params, n_images=3, img_size=128):
    """Generate feature map visualization."""
    names = ["Skull", "Butterfly", "Apple"][:n_images]
    features_list = []

    for img_id in range(n_images):
        _, features = cppn.generate_image(params, image_id=img_id, img_size=img_size, return_features=True)
        features = jax.tree.map(lambda x: np.array(x), features)
        features_list.append(features)

    # Get dimensions
    features_0 = features_list[0]
    max_features_per_layer = max(jax.tree.map(lambda x: x.shape[-1], features_0))
    n_layers = len(features_0)

    fig = plt.figure(figsize=(6.5 * n_images, n_layers/max_features_per_layer*6.5), dpi=100)

    for img_idx in range(n_images):
        features = features_list[img_idx]

        for i in range(n_layers):
            layer_features = jnp.transpose(features[i], (2, 0, 1))

            for j in range(max_features_per_layer):
                subplot_idx = img_idx * max_features_per_layer + j + 1
                ax = plt.subplot(n_layers, max_features_per_layer * n_images,
                               (n_layers-1-i) * max_features_per_layer * n_images + subplot_idx)

                if j >= layer_features.shape[0]:
                    ax.set_visible(False)
                    continue

                fmap = layer_features[j]
                plt.imshow(fmap, cmap='bwr_r', vmin=-1.0, vmax=1.0)
                plt.xticks([])
                plt.yticks([])

                if j == 0 and i % 2 == 0:
                    plt.ylabel(f"{i}", fontsize=8)

                if i == n_layers - 1:
                    if j == max_features_per_layer // 2:
                        plt.title(f"{names[img_idx]}", fontsize=12, fontweight='bold')

                for spine in ax.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(0.1)

    plt.subplots_adjust(left=0.06, right=1., bottom=0.05, top=0.95, wspace=0.15, hspace=0.15)
    fig.supylabel("Layer", fontsize=12, x=0.)
    fig.supxlabel("Neuron", fontsize=12, y=0.0)
    plt.suptitle("Feature Maps", fontsize=16, fontweight='bold', y=0.98)

    return fig


def run_all_visualizations(cppn, params, save_dir, n_images=3):
    """
    Run all post-training visualizations and save to save_dir.

    Args:
        cppn: FlattenConditionalCPPNParameters wrapper
        params: Flattened parameters
        save_dir: Directory to save visualizations
        n_images: Number of images in the conditional model
    """
    print("\n" + "="*60)
    print("GENERATING POST-TRAINING VISUALIZATIONS")
    print("="*60)

    # Edge interpolations
    print("Generating edge interpolations...")
    fig = generate_edge_interpolations(cppn, params, n_images=n_images, img_size=128)
    plt.savefig(f"{save_dir}/edge_interpolations.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir}/edge_interpolations.png")

    # Simplex sampling
    print("Generating simplex sampling...")
    fig = generate_simplex_sampling(cppn, params, n_images=n_images, img_size=64)
    plt.savefig(f"{save_dir}/simplex_sampling.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir}/simplex_sampling.png")

    # Feature maps
    print("Generating feature maps...")
    fig = generate_feature_maps(cppn, params, n_images=n_images, img_size=128)
    plt.savefig(f"{save_dir}/feature_maps.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_dir}/feature_maps.png")

    print("="*60 + "\n")
