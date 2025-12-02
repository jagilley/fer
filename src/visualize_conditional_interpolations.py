"""
Visualize conditional CPPN outputs at interpolated conditioning values.

This explores what the network generates between the discrete training points,
which can reveal FER vs UFR characteristics:
- FER: Interpolations are incoherent because separate circuits don't blend smoothly
- UFR: Interpolations are meaningful because shared features blend smoothly
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import jax
import jax.numpy as jnp
from functools import partial

from cppn_conditional import ConditionalCPPN, FlattenConditionalCPPNParameters
import util


def generate_at_condition(cppn, params, condition, img_size=128):
    """
    Generate an image at an arbitrary conditioning vector (doesn't have to be one-hot).

    Args:
        cppn: The conditional CPPN wrapper
        params: Flattened parameters
        condition: Array of shape (n_images,) with conditioning values (e.g., [0.5, 0.5, 0])
        img_size: Image resolution
    """
    # Generate coordinate grid
    x = y = jnp.linspace(-1, 1, img_size)
    grid_x, grid_y = jnp.meshgrid(x, y, indexing='ij')

    # Create inputs dict
    inputs = {}
    inputs['x'] = grid_x
    inputs['y'] = grid_y
    inputs['d'] = jnp.sqrt(grid_x**2 + grid_y**2) * 1.4
    inputs['b'] = jnp.ones_like(grid_x)

    # Add conditioning as constant values (not one-hot, but arbitrary weights)
    for i in range(len(condition)):
        inputs[f'img_{i}'] = jnp.full_like(grid_x, condition[i])

    # Construct input vector
    base_inputs = [inputs[name] for name in cppn.cppn.inputs.split(",")]
    cond_inputs = [inputs[f'img_{i}'] for i in range(len(condition))]
    all_inputs = base_inputs + cond_inputs
    inputs_stacked = jnp.stack(all_inputs, axis=-1)

    # Generate image
    from cppn import hsv2rgb
    (h, s, v), _ = jax.vmap(jax.vmap(partial(cppn.cppn.apply, params)))(inputs_stacked)
    r, g, b = hsv2rgb((h+1)%1, s.clip(0,1), jnp.abs(v).clip(0, 1))
    rgb = jnp.stack([r, g, b], axis=-1)

    return np.array(rgb)


def visualize_edge_interpolations(cppn, params, img_size=128):
    """
    Visualize interpolations along the edges of the simplex.
    """
    fig, axes = plt.subplots(3, 7, figsize=(21, 9))

    edges = [
        ("Skull", "Butterfly", 0, 1),
        ("Skull", "Apple", 0, 2),
        ("Butterfly", "Apple", 1, 2)
    ]

    for edge_idx, (name1, name2, idx1, idx2) in enumerate(edges):
        # Generate 7 interpolations along this edge
        alphas = np.linspace(0, 1, 7)

        for col_idx, alpha in enumerate(alphas):
            condition = np.zeros(3)
            condition[idx1] = 1 - alpha
            condition[idx2] = alpha

            img = generate_at_condition(cppn, params, condition, img_size)

            axes[edge_idx, col_idx].imshow(img)
            axes[edge_idx, col_idx].axis('off')

            if edge_idx == 0:
                if col_idx == 0:
                    axes[edge_idx, col_idx].set_title(f"{name1}\n[1,0,0]", fontsize=10)
                elif col_idx == 6:
                    axes[edge_idx, col_idx].set_title(f"{name2}\n[0,1,0]", fontsize=10)
                else:
                    axes[edge_idx, col_idx].set_title(f"α={alpha:.2f}", fontsize=9)

            # Add row labels
            if col_idx == 0:
                axes[edge_idx, col_idx].text(-0.1, 0.5, f"{name1}↔{name2}",
                                              transform=axes[edge_idx, col_idx].transAxes,
                                              fontsize=12, fontweight='bold',
                                              rotation=90, va='center', ha='right')

    plt.suptitle("Edge Interpolations: Blending Between Training Points",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig


def visualize_simplex_sampling(cppn, params, img_size=64):
    """
    Sample points throughout the 2-simplex (triangle) of conditioning values.
    """
    # Create a triangular grid of samples
    # We'll use barycentric coordinates

    n_levels = 5  # Number of levels in the triangle
    samples = []

    for i in range(n_levels):
        for j in range(n_levels - i):
            k = n_levels - 1 - i - j
            # Barycentric coordinates (sum to n_levels-1)
            w0, w1, w2 = i, j, k
            # Normalize to sum to 1
            condition = np.array([w0, w1, w2]) / (n_levels - 1)
            samples.append(condition)

    # Generate images for all samples
    print(f"Generating {len(samples)} interpolated samples...")
    images = []
    for i, condition in enumerate(samples):
        img = generate_at_condition(cppn, params, condition, img_size)
        images.append(img)

    # Create visualization in triangular layout
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.axis('off')

    # Plot images in triangular arrangement
    img_idx = 0
    for i in range(n_levels):
        for j in range(n_levels - i):
            k = n_levels - 1 - i - j
            w0, w1, w2 = i, j, k
            condition = np.array([w0, w1, w2]) / (n_levels - 1)

            # Convert barycentric to 2D position
            # Skull at top (0, 1), Butterfly at bottom-left (0, 0), Apple at bottom-right (1, 0)
            x = condition[2] + condition[0] * 0.5
            y = condition[0] * np.sqrt(3) / 2

            img = images[img_idx]
            img_idx += 1

            # Plot image
            extent = [x - 0.08, x + 0.08, y - 0.08, y + 0.08]
            ax.imshow(img, extent=extent, zorder=10)

            # Add border
            from matplotlib.patches import Rectangle
            rect = Rectangle((x - 0.08, y - 0.08), 0.16, 0.16,
                           linewidth=1.5, edgecolor='black', facecolor='none', zorder=11)
            ax.add_patch(rect)

    # Draw simplex triangle
    triangle = plt.Polygon([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]],
                          fill=False, edgecolor='gray', linewidth=2, linestyle='--', zorder=1)
    ax.add_patch(triangle)

    # Label vertices
    ax.text(0, 0, "Butterfly\n[0,1,0]", ha='center', va='top', fontsize=14, fontweight='bold')
    ax.text(1, 0, "Apple\n[0,0,1]", ha='center', va='top', fontsize=14, fontweight='bold')
    ax.text(0.5, np.sqrt(3)/2, "Skull\n[1,0,0]", ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.title("2-Simplex: Conditional CPPN Throughout Conditioning Space",
             fontsize=16, fontweight='bold', pad=20)

    return fig


def compute_interpolation_smoothness(cppn, params, n_samples=20, img_size=64):
    """
    Compute a smoothness metric for interpolations.

    Measures the average pixel-wise variance along interpolation paths.
    High variance suggests discontinuous/fractured representations.
    Low variance suggests smooth/factored representations.
    """
    print("\n" + "="*60)
    print("INTERPOLATION SMOOTHNESS ANALYSIS")
    print("="*60)
    print("High smoothness → UFR (features blend coherently)")
    print("Low smoothness → FER (abrupt transitions, incoherence)")
    print("-"*60)

    edges = [
        ("Skull → Butterfly", 0, 1),
        ("Skull → Apple", 0, 2),
        ("Butterfly → Apple", 1, 2)
    ]

    smoothness_scores = []

    for name, idx1, idx2 in edges:
        # Generate n_samples along this edge
        alphas = np.linspace(0, 1, n_samples)
        imgs = []

        for alpha in alphas:
            condition = np.zeros(3)
            condition[idx1] = 1 - alpha
            condition[idx2] = alpha
            img = generate_at_condition(cppn, params, condition, img_size)
            imgs.append(img)

        imgs = np.array(imgs)

        # Compute pixel-wise differences between adjacent samples
        diffs = np.diff(imgs, axis=0)
        avg_diff = np.mean(np.abs(diffs))
        max_diff = np.max(np.abs(diffs))

        # Compute variance along the path
        variance = np.var(imgs, axis=0).mean()

        # Smoothness score: inverse of average difference (normalized)
        smoothness = 1.0 / (1.0 + avg_diff * 10)
        smoothness_scores.append(smoothness)

        print(f"\n{name}:")
        print(f"  Average pixel change: {avg_diff:.4f}")
        print(f"  Max pixel change: {max_diff:.4f}")
        print(f"  Path variance: {variance:.4f}")
        print(f"  Smoothness score: {smoothness:.4f}")

    overall_smoothness = np.mean(smoothness_scores)
    print("\n" + "="*60)
    print(f"OVERALL SMOOTHNESS: {overall_smoothness:.4f}")
    if overall_smoothness > 0.7:
        print("→ High smoothness: Suggests UFR-like blending")
    elif overall_smoothness > 0.5:
        print("→ Moderate smoothness: Mixed representation")
    else:
        print("→ Low smoothness: Suggests FER with abrupt transitions")
    print("="*60 + "\n")

    return smoothness_scores


def main(save_dir=None):
    # Load the trained conditional CPPN
    if save_dir is None:
        save_dir = "../data/conditional_small_direct"

    print(f"Loading conditional CPPN from: {save_dir}")
    arch = util.load_pkl(save_dir, "arch")
    params = util.load_pkl(save_dir, "params")
    args = util.load_pkl(save_dir, "args")

    print(f"Architecture: {arch}")
    print(f"Number of images: {args.n_images}")

    # Create CPPN
    cppn = FlattenConditionalCPPNParameters(ConditionalCPPN(arch=arch, n_images=args.n_images))
    print(f"Total parameters: {len(params)}")

    # Unflatten params for use with the CPPN
    params_unflattened = cppn.param_reshaper.reshape_single(params)

    # 1. Edge interpolations
    print("\n" + "="*60)
    print("Generating edge interpolations...")
    print("="*60)
    fig1 = visualize_edge_interpolations(cppn, params_unflattened, img_size=128)
    output_path_1 = f"{save_dir}/edge_interpolations.png"
    plt.savefig(output_path_1, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path_1}")
    plt.close()

    # 2. Simplex sampling
    print("\n" + "="*60)
    print("Generating simplex sampling...")
    print("="*60)
    fig2 = visualize_simplex_sampling(cppn, params_unflattened, img_size=64)
    output_path_2 = f"{save_dir}/simplex_sampling.png"
    plt.savefig(output_path_2, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path_2}")
    plt.close()

    # 3. Smoothness analysis
    smoothness_scores = compute_interpolation_smoothness(cppn, params_unflattened, n_samples=20, img_size=64)

    print("\nAnalysis complete!")
    print("\nKey insights:")
    print("- Edge interpolations show how the model blends between training points")
    print("- Simplex sampling reveals the full conditioning space structure")
    print("- Smoothness scores quantify interpolation quality (relevant for distillation)")
    print("\nFor distillation:")
    print("- Smooth interpolations → teacher can provide meaningful supervision everywhere")
    print("- Abrupt transitions → distillation may be harder in those regions")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(save_dir=sys.argv[1])
    else:
        main()
