"""
FER/UFR metrics for analyzing conditional CPPNs during and after training.

This module provides modular metric computation functions that can be used
to track representation quality throughout training.
"""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial


def compute_feature_similarity(cppn, params, n_images=3, img_size=64):
    """
    Compute average correlation between feature maps across all image pairs.

    Returns:
        float: Average correlation (0-1). Higher = more UFR, lower = more FER.
    """
    # Generate features for each image
    features_list = []
    for img_id in range(n_images):
        _, features = cppn.generate_image(params, image_id=img_id, img_size=img_size, return_features=True)
        features_list.append(features)

    # Compute correlations between all pairs
    n_layers = len(features_list[0])
    layer_correlations = []

    for layer_idx in range(n_layers):
        layer_corrs = []
        for i in range(n_images):
            for j in range(i+1, n_images):
                feat_i = features_list[i][layer_idx].flatten()
                feat_j = features_list[j][layer_idx].flatten()
                corr = np.corrcoef(feat_i, feat_j)[0, 1]
                if not np.isnan(corr):  # Handle edge cases
                    layer_corrs.append(corr)

        if layer_corrs:
            layer_correlations.append(np.mean(layer_corrs))

    # Return average across all layers
    return np.mean(layer_correlations) if layer_correlations else 0.0


def compute_spatial_roughness(cppn, params, n_images=3, img_size=64):
    """
    Compute average spatial roughness of feature maps.

    Roughness measures how much adjacent pixels differ - smooth gradients have low
    roughness, noisy/patchy patterns have high roughness. Picbreeder (UFR) CPPNs
    have dramatically lower roughness than SGD-trained (FER) CPPNs.

    This is the most direct measure of "clean functional composition" - UFR networks
    compute smooth transformations, while FER networks develop noisy/patchy patterns.

    Returns:
        dict with:
            - avg_roughness: Mean roughness across all feature maps (lower = more UFR)
            - max_roughness: Maximum roughness across all feature maps (lower = more UFR)
    """
    # Generate features for each image
    features_list = []
    for img_id in range(n_images):
        _, features = cppn.generate_image(params, image_id=img_id, img_size=img_size, return_features=True)
        features_list.append([np.array(f) for f in features])

    all_roughnesses = []

    n_layers = len(features_list[0])
    for layer_idx in range(n_layers):
        for img_id in range(n_images):
            layer_feat = features_list[img_id][layer_idx]
            n_neurons = layer_feat.shape[-1]

            for n in range(n_neurons):
                fmap = layer_feat[:, :, n]

                # Compute gradient magnitude (roughness)
                gx = np.diff(fmap, axis=0)
                gy = np.diff(fmap, axis=1)
                roughness = (np.mean(np.abs(gx)) + np.mean(np.abs(gy))) / 2
                all_roughnesses.append(roughness)

    return {
        "avg_roughness": np.mean(all_roughnesses),
        "max_roughness": np.max(all_roughnesses),
    }


def compute_neuron_specialization(cppn, params, n_images=3, img_size=64):
    """
    Compute average neuron specialization (coefficient of variation).

    Returns:
        float: Average CV across neurons. Higher = more FER, lower = more UFR.
    """
    # Generate features for each image
    features_list = []
    for img_id in range(n_images):
        _, features = cppn.generate_image(params, image_id=img_id, img_size=img_size, return_features=True)
        features_list.append(features)

    # Compute specialization for each neuron
    n_layers = len(features_list[0])
    all_cvs = []

    for layer_idx in range(n_layers):
        layer_features = [features_list[i][layer_idx] for i in range(n_images)]
        n_neurons = layer_features[0].shape[-1]

        for neuron_idx in range(n_neurons):
            # Get mean activation for this neuron across each image
            activations = [np.abs(feat[:, :, neuron_idx]).mean() for feat in layer_features]
            mean_act = np.mean(activations)

            if mean_act > 1e-8:  # Only consider active neurons
                variance = np.var(activations)
                cv = np.sqrt(variance) / mean_act
                all_cvs.append(cv)

    # Return average CV across all neurons
    return np.mean(all_cvs) if all_cvs else 0.0


def compute_interpolation_smoothness(cppn, params, n_images=3, n_samples=10, img_size=32):
    """
    Compute smoothness of interpolations along edges of the conditioning simplex.

    Args:
        cppn: FlattenConditionalCPPNParameters wrapper
        params: Flat parameter array
        n_images: Number of images
        n_samples: Number of interpolation points
        img_size: Image resolution for interpolations

    Returns:
        float: Average smoothness (0-1). Higher = more UFR, lower = more FER.
    """
    # Simply use the cppn.generate_image method which handles param format
    edges = [(0, 1), (0, 2), (1, 2)]
    all_smoothness = []

    for idx1, idx2 in edges:
        alphas = np.linspace(0, 1, n_samples)
        imgs = []

        for alpha in alphas:
            # Create conditioning
            condition = np.zeros(n_images)
            condition[idx1] = 1 - alpha
            condition[idx2] = alpha

            # Use generate_at_condition helper
            img = _generate_at_condition(cppn, params, condition, img_size)
            imgs.append(np.array(img))

        imgs = np.array(imgs)

        # Compute smoothness
        diffs = np.diff(imgs, axis=0)
        avg_diff = np.mean(np.abs(diffs))
        smoothness = 1.0 / (1.0 + avg_diff * 10)
        all_smoothness.append(smoothness)

    return np.mean(all_smoothness)


def _generate_at_condition(cppn, params, condition, img_size):
    """Helper to generate image at arbitrary conditioning (not necessarily one-hot)."""
    from cppn import hsv2rgb

    # Unflatten params for use with cppn.cppn.apply
    params_unflattened = cppn.param_reshaper.reshape_single(params)

    # Generate coordinate grid
    x = y = jnp.linspace(-1, 1, img_size)
    grid_x, grid_y = jnp.meshgrid(x, y, indexing='ij')

    # Create inputs
    inputs = {}
    inputs['x'] = grid_x
    inputs['y'] = grid_y
    inputs['d'] = jnp.sqrt(grid_x**2 + grid_y**2) * 1.4
    inputs['b'] = jnp.ones_like(grid_x)

    for i in range(len(condition)):
        inputs[f'img_{i}'] = jnp.full_like(grid_x, condition[i])

    base_inputs = [inputs[name] for name in cppn.cppn.inputs.split(",")]
    cond_inputs = [inputs[f'img_{i}'] for i in range(len(condition))]
    all_inputs = base_inputs + cond_inputs
    inputs_stacked = jnp.stack(all_inputs, axis=-1)

    (h, s, v), _ = jax.vmap(jax.vmap(partial(cppn.cppn.apply, params_unflattened)))(inputs_stacked)
    r, g, b = hsv2rgb((h+1)%1, s.clip(0,1), jnp.abs(v).clip(0, 1))
    rgb = jnp.stack([r, g, b], axis=-1)

    return rgb


def compute_all_metrics(cppn, params, n_images=3, img_size_features=64, img_size_interp=32):
    """
    Compute all FER/UFR metrics at once.

    Returns:
        dict: Dictionary with all metric values
    """
    roughness = compute_spatial_roughness(cppn, params, n_images, img_size_features)
    return {
        "feature_similarity": compute_feature_similarity(cppn, params, n_images, img_size_features),
        "neuron_specialization": compute_neuron_specialization(cppn, params, n_images, img_size_features),
        "interpolation_smoothness": compute_interpolation_smoothness(cppn, params, n_images, img_size=img_size_interp),
        "spatial_roughness": roughness["avg_roughness"],
        "max_roughness": roughness["max_roughness"],
    }


def print_metrics(metrics, step=None):
    """Pretty print metrics."""
    if step is not None:
        print(f"\nStep {step}:")
    else:
        print("\nMetrics:")

    print(f"  Feature Similarity:       {metrics['feature_similarity']:.4f} (UFR: high, FER: low)")
    print(f"  Neuron Specialization:    {metrics['neuron_specialization']:.4f} (UFR: low, FER: high)")
    print(f"  Interpolation Smoothness: {metrics['interpolation_smoothness']:.4f} (UFR: high, FER: low)")
    print(f"  Spatial Roughness:        {metrics['spatial_roughness']:.4f} (UFR: low, FER: high)")
    print(f"  Max Roughness:            {metrics['max_roughness']:.4f} (UFR: <0.10, FER: >0.15)")
