"""
Visualize feature maps for the conditional CPPN to analyze FER vs UFR characteristics.

This script generates feature maps for each image_id and visualizes them side-by-side
to determine if the model uses:
- FER (Fractured Entangled Representation): Different circuits for each image
- UFR (Unified Factored Representation): Shared structure across images
"""

import os
import sys
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import jax
import jax.numpy as jnp

from cppn_conditional import ConditionalCPPN, FlattenConditionalCPPNParameters
import util

def viz_feature_maps_conditional(features_list, image_names, title="Conditional CPPN Feature Maps"):
    """
    Visualize feature maps for multiple images side-by-side.

    Args:
        features_list: List of features (one per image_id)
        image_names: List of image names (e.g., ["Skull", "Butterfly", "Apple"])
        title: Overall title for the figure
    """
    n_images = len(features_list)

    # Get dimensions
    features_0 = features_list[0]
    max_features_per_layer = max(jax.tree.map(lambda x: x.shape[-1], features_0))
    n_layers = len(features_0)

    # Create a figure with subplots for each image
    fig = plt.figure(figsize=(6.5 * n_images, n_layers/max_features_per_layer*6.5), dpi=150)

    for img_idx in range(n_images):
        features = features_list[img_idx]

        # Create subplot grid for this image
        for i in tqdm(range(n_layers), desc=f"Processing {image_names[img_idx]}"):
            layer_features = jnp.transpose(features[i], (2, 0, 1))  # D H W

            for j in range(max_features_per_layer):
                # Calculate global subplot position
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

                # Add labels
                if j == 0 and i % 2 == 0:
                    plt.ylabel(f"{i}", fontsize=8)

                if i == 0:
                    if img_idx == 0:
                        plt.xlabel(["$x$", "$y$", "$d$", "$b$"][j] if j < 4 else f"${j}$", fontsize=8)
                    else:
                        plt.xlabel(f"${j}$" if j >= 4 else "", fontsize=8)

                if i == n_layers - 1:
                    if j == max_features_per_layer // 2:
                        plt.title(f"{image_names[img_idx]}", fontsize=12, fontweight='bold')

                for spine in ax.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(0.1)

    plt.subplots_adjust(left=0.06, right=1., bottom=0.05, top=0.95, wspace=0.15, hspace=0.15)
    fig.supylabel("Layer", fontsize=12, x=0.)
    fig.supxlabel("Neuron", fontsize=12, y=0.0)
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    return fig


def analyze_feature_similarity(features_list, image_names):
    """
    Analyze similarity of feature maps across different images.
    High similarity suggests UFR, low similarity suggests FER.
    """
    n_images = len(features_list)
    n_layers = len(features_list[0])

    print("\n" + "="*60)
    print("FEATURE MAP SIMILARITY ANALYSIS")
    print("="*60)
    print("High correlation → UFR (shared representations)")
    print("Low correlation → FER (separate circuits)")
    print("-"*60)

    similarities = []

    for layer_idx in range(n_layers):
        print(f"\nLayer {layer_idx}:")
        layer_sims = []

        for i in range(n_images):
            for j in range(i+1, n_images):
                feat_i = features_list[i][layer_idx].flatten()
                feat_j = features_list[j][layer_idx].flatten()

                # Compute correlation
                corr = np.corrcoef(feat_i, feat_j)[0, 1]
                layer_sims.append(corr)

                print(f"  {image_names[i]} vs {image_names[j]}: correlation = {corr:.4f}")

        avg_sim = np.mean(layer_sims)
        similarities.append(avg_sim)
        print(f"  Average correlation: {avg_sim:.4f}")

    overall_avg = np.mean(similarities)
    print("\n" + "="*60)
    print(f"OVERALL AVERAGE CORRELATION: {overall_avg:.4f}")
    if overall_avg < 0.3:
        print("→ Strong FER: Networks use mostly separate circuits")
    elif overall_avg < 0.6:
        print("→ Moderate FER: Networks share some but not all structure")
    else:
        print("→ UFR-like: Networks share significant structure")
    print("="*60 + "\n")

    return similarities


def analyze_neuron_specialization(features_list, image_names):
    """
    Analyze if neurons specialize to specific images (FER) or generalize (UFR).
    """
    print("\n" + "="*60)
    print("NEURON SPECIALIZATION ANALYSIS")
    print("="*60)
    print("High specialization → FER (neurons active for specific images)")
    print("Low specialization → UFR (neurons active across images)")
    print("-"*60)

    n_layers = len(features_list[0])

    for layer_idx in range(n_layers):
        print(f"\nLayer {layer_idx}:")

        # Get feature maps for this layer from all images
        layer_features = [features_list[i][layer_idx] for i in range(len(features_list))]

        # For each neuron, compute variance across images
        n_neurons = layer_features[0].shape[-1]

        for neuron_idx in range(min(n_neurons, 5)):  # Show first 5 neurons
            activations = [np.abs(feat[:, :, neuron_idx]).mean() for feat in layer_features]
            variance = np.var(activations)
            mean_act = np.mean(activations)

            # Compute specialization score (coefficient of variation)
            cv = np.sqrt(variance) / (mean_act + 1e-8)

            print(f"  Neuron {neuron_idx}: activations={[f'{a:.3f}' for a in activations]}, CV={cv:.3f}")

    print("="*60 + "\n")


def main():
    # Load the trained conditional CPPN
    save_dir = "../data/conditional_small_direct"

    print("Loading conditional CPPN...")
    arch = util.load_pkl(save_dir, "arch")
    params = util.load_pkl(save_dir, "params")
    args = util.load_pkl(save_dir, "args")

    print(f"Architecture: {arch}")
    print(f"Number of images: {args.n_images}")

    # Create CPPN
    cppn = FlattenConditionalCPPNParameters(ConditionalCPPN(arch=arch, n_images=args.n_images))
    print(f"Total parameters: {len(params)}")

    # Generate feature maps for each image
    image_names = ["Skull", "Butterfly", "Apple"]
    features_list = []
    imgs_list = []

    print("\nGenerating feature maps for each image...")
    for img_id in range(args.n_images):
        print(f"  Generating for {image_names[img_id]} (image_id={img_id})...")
        img, features = cppn.generate_image(params, image_id=img_id, img_size=128, return_features=True)
        features = jax.tree.map(lambda x: np.array(x), features)
        features_list.append(features)
        imgs_list.append(np.array(img))

    # Analyze feature similarity
    analyze_feature_similarity(features_list, image_names)

    # Analyze neuron specialization
    analyze_neuron_specialization(features_list, image_names)

    # Create visualization
    print("Creating feature map visualization...")
    fig = viz_feature_maps_conditional(
        features_list,
        image_names,
        title="Conditional CPPN Feature Maps (Direct Training, Expecting FER)"
    )

    # Save figure
    output_path = "conditional_feature_maps.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved feature map visualization to: {output_path}")

    # Also create a simple comparison of the output images
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (img, name) in enumerate(zip(imgs_list, image_names)):
        axes[i].imshow(img)
        axes[i].set_title(f"{name}", fontsize=16, fontweight='bold')
        axes[i].axis('off')
    plt.suptitle("Conditional CPPN Outputs", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig("conditional_outputs_comparison.png", dpi=150, bbox_inches='tight')
    print(f"Saved output comparison to: conditional_outputs_comparison.png")

    print("\nAnalysis complete!")
    print("\nInterpretation Guide:")
    print("- FER signatures: Low correlation, high specialization, visually distinct feature maps")
    print("- UFR signatures: High correlation, low specialization, similar feature maps with variations")


if __name__ == "__main__":
    main()
