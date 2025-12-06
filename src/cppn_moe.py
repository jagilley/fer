"""
Mixture-of-Experts Conditional CPPN.

Hypothesis: MoE models achieve better (more UFR-like) representations than dense models
because parameter-limited experts need to generalize more, while in aggregate they
have equivalent coverage to the dense model.

Key design: Router uses spatial coords (x,y,d,b) only, NOT image_id.
This forces routing to be image-agnostic, so each expert must learn
representations that work across all images.
"""

from functools import partial
import numpy as np

import jax
import jax.numpy as jnp
from jax.random import split
import flax
import flax.linen as nn

from einops import rearrange

import evosax

from color import hsv2rgb

# Activation functions (same as cppn_conditional.py)
cache = lambda x: x
identity = lambda x: x
cos = jnp.cos
sin = jnp.sin
tanh = jnp.tanh
sigmoid = lambda x: jax.nn.sigmoid(x) * 2. - 1.
gaussian = lambda x: jnp.exp(-x**2) * 2. - 1.
relu = jax.nn.relu
activation_fn_map = dict(cache=cache, identity=identity, cos=cos, sin=sin, tanh=tanh, sigmoid=sigmoid, gaussian=gaussian, relu=relu)


class ExpertCPPN(nn.Module):
    """
    A single expert CPPN (same architecture as tiny CPPN).
    """
    arch: str = "6;cache:8,gaussian:2,identity:1,sin:1"

    @nn.compact
    def __call__(self, x):
        n_layers, activation_neurons = self.arch.split(";")
        n_layers = int(n_layers)

        activations = [i.split(":")[0] for i in activation_neurons.split(",")]
        d_hidden = [int(i.split(":")[-1]) for i in activation_neurons.split(",")]
        dh_cumsum = list(np.cumsum(d_hidden))

        features = [x]
        for i_layer in range(n_layers):
            x = nn.Dense(sum(d_hidden), use_bias=False)(x)
            x = jnp.split(x, dh_cumsum)
            x = [activation_fn_map[activation](xi) for xi, activation in zip(x, activations)]
            x = jnp.concatenate(x)
            features.append(x)

        x = nn.Dense(3, use_bias=False)(x)
        features.append(x)
        h, s, v = x
        return (h, s, v), features


class Router(nn.Module):
    """
    Router network that outputs expert weights based on spatial coordinates only.

    IMPORTANT: Router input is (x, y, d, b) only - NOT image_id.
    This ensures experts can't specialize per-image.
    """
    n_experts: int = 6
    hidden_dim: int = 48

    @nn.compact
    def __call__(self, spatial_coords):
        """
        Args:
            spatial_coords: (x, y, d, b) - shape (..., 4)
        Returns:
            Expert weights - shape (..., n_experts), softmax normalized
        """
        x = nn.Dense(self.hidden_dim, use_bias=False)(spatial_coords)
        x = nn.tanh(x)
        x = nn.Dense(self.n_experts, use_bias=False)(x)
        weights = nn.softmax(x, axis=-1)
        return weights


class MoEConditionalCPPN(nn.Module):
    """
    Mixture-of-Experts Conditional CPPN.

    Architecture:
    - Router: Takes spatial coords (x,y,d,b), outputs expert weights
    - Experts: K tiny CPPNs, each receives full input (x,y,d,b,image_id)
    - Output: Weighted sum of expert outputs

    Key insight: Router is image-agnostic, so experts must learn
    general representations that work across all images.

    Args:
        expert_arch: Architecture string for each expert (tiny CPPN)
        n_experts: Number of experts
        n_images: Number of images (for one-hot conditioning)
        router_hidden: Hidden dimension of router network
        inputs: Base spatial inputs
    """
    expert_arch: str = "6;cache:8,gaussian:2,identity:1,sin:1"
    n_experts: int = 6
    n_images: int = 3
    router_hidden: int = 48
    inputs: str = "x,y,d,b"

    def setup(self):
        self.router = Router(n_experts=self.n_experts, hidden_dim=self.router_hidden)
        self.experts = [ExpertCPPN(arch=self.expert_arch) for _ in range(self.n_experts)]

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: Full input including one-hot image_id (..., 4 + n_images)
        Returns:
            (h, s, v), features
        """
        # Split input into spatial coords and image_id
        n_spatial = len(self.inputs.split(","))  # 4 for x,y,d,b
        spatial_coords = x[..., :n_spatial]

        # Router outputs weights based on spatial coords only
        expert_weights = self.router(spatial_coords)  # (..., n_experts)

        # Run all experts on full input
        expert_outputs = []
        expert_features = []
        for expert in self.experts:
            (h, s, v), feats = expert(x)
            expert_outputs.append(jnp.stack([h, s, v], axis=-1))  # (..., 3)
            expert_features.append(feats)

        expert_outputs = jnp.stack(expert_outputs, axis=-2)  # (..., n_experts, 3)

        # Weighted combination of expert outputs
        # expert_weights: (..., n_experts) -> (..., n_experts, 1)
        weights_expanded = expert_weights[..., None]
        combined = jnp.sum(expert_outputs * weights_expanded, axis=-2)  # (..., 3)

        h, s, v = combined[..., 0], combined[..., 1], combined[..., 2]

        # For features, we'll aggregate across experts (weighted average)
        # This gives us a single set of features to analyze
        n_layers = len(expert_features[0])
        combined_features = []
        for layer_idx in range(n_layers):
            layer_feats = jnp.stack([expert_features[i][layer_idx] for i in range(self.n_experts)], axis=-2)
            # Weighted average of features across experts
            combined_layer = jnp.sum(layer_feats * expert_weights[..., None], axis=-2)
            combined_features.append(combined_layer)

        return (h, s, v), combined_features

    def generate_image(self, params, image_id, img_size=256, return_features=False):
        """
        Generate an image from the MoE conditional CPPN.
        """
        inputs = {}
        x = y = jnp.linspace(-1, 1, img_size)
        inputs['x'], inputs['y'] = jnp.meshgrid(x, y, indexing='ij')
        inputs['d'] = jnp.sqrt(inputs['x']**2 + inputs['y']**2) * 1.4
        inputs['b'] = jnp.ones_like(inputs['x'])

        # Add image_id as one-hot encoding
        image_id_onehot = jnp.eye(self.n_images)[image_id]
        for i in range(self.n_images):
            inputs[f'img_{i}'] = jnp.full_like(inputs['x'], image_id_onehot[i])

        # Construct input vector: base inputs + one-hot image_id
        base_inputs = [inputs[input_name] for input_name in self.inputs.split(",")]
        img_id_inputs = [inputs[f'img_{i}'] for i in range(self.n_images)]
        all_inputs = base_inputs + img_id_inputs
        inputs_stacked = jnp.stack(all_inputs, axis=-1)

        (h, s, v), features = jax.vmap(jax.vmap(partial(self.apply, params)))(inputs_stacked)
        r, g, b = hsv2rgb((h+1)%1, s.clip(0,1), jnp.abs(v).clip(0, 1))
        rgb = jnp.stack([r, g, b], axis=-1)

        if return_features:
            return rgb, features
        else:
            return rgb

    def generate_image_with_routing(self, params, image_id, img_size=256):
        """
        Generate an image and also return the routing weights for visualization.
        """
        inputs = {}
        x = y = jnp.linspace(-1, 1, img_size)
        inputs['x'], inputs['y'] = jnp.meshgrid(x, y, indexing='ij')
        inputs['d'] = jnp.sqrt(inputs['x']**2 + inputs['y']**2) * 1.4
        inputs['b'] = jnp.ones_like(inputs['x'])

        image_id_onehot = jnp.eye(self.n_images)[image_id]
        for i in range(self.n_images):
            inputs[f'img_{i}'] = jnp.full_like(inputs['x'], image_id_onehot[i])

        base_inputs = [inputs[input_name] for input_name in self.inputs.split(",")]
        img_id_inputs = [inputs[f'img_{i}'] for i in range(self.n_images)]
        all_inputs = base_inputs + img_id_inputs
        inputs_stacked = jnp.stack(all_inputs, axis=-1)

        # Get router weights
        spatial_coords = inputs_stacked[..., :4]

        def forward_with_routing(params, x):
            (h, s, v), features = self.apply(params, x)
            # Also get routing weights
            router_weights = self.apply(params, x[..., :4], method=lambda m, sc: m.router(sc))
            return (h, s, v), router_weights

        # Simpler approach: just run forward and extract routing
        (h, s, v), features = jax.vmap(jax.vmap(partial(self.apply, params)))(inputs_stacked)
        r, g, b = hsv2rgb((h+1)%1, s.clip(0,1), jnp.abs(v).clip(0, 1))
        rgb = jnp.stack([r, g, b], axis=-1)

        # Get routing weights separately
        router_params = {'params': {k: v for k, v in params['params'].items() if 'router' in k.lower() or k == 'Router_0'}}
        # Actually, let's compute routing in a simpler way
        routing_weights = jax.vmap(jax.vmap(
            lambda sc: nn.softmax(
                nn.Dense(self.n_experts, use_bias=False).apply(
                    {'params': params['params']['Router_0']['Dense_1']},
                    nn.tanh(nn.Dense(self.router_hidden, use_bias=False).apply(
                        {'params': params['params']['Router_0']['Dense_0']}, sc
                    ))
                )
            )
        ))(spatial_coords)

        return rgb, routing_weights


class FlattenMoECPPNParameters():
    """
    Flatten the parameters of the MoE CPPN to a single vector.
    """
    def __init__(self, cppn):
        self.cppn = cppn

        rng = jax.random.PRNGKey(0)
        # Input dimensionality: base inputs + n_images for one-hot
        d_in = len(self.cppn.inputs.split(",")) + self.cppn.n_images
        self.param_reshaper = evosax.ParameterReshaper(self.cppn.init(rng, jnp.zeros((d_in,))))
        self.n_params = self.param_reshaper.total_params

    def init(self, rng):
        d_in = len(self.cppn.inputs.split(",")) + self.cppn.n_images
        params = self.cppn.init(rng, jnp.zeros((d_in,)))
        return self.param_reshaper.flatten_single(params)

    def generate_image(self, params, image_id, img_size=256, return_features=False):
        params = self.param_reshaper.reshape_single(params)
        return self.cppn.generate_image(params, image_id=image_id, img_size=img_size, return_features=return_features)

    def generate_image_with_routing(self, params, image_id, img_size=256):
        params = self.param_reshaper.reshape_single(params)
        return self.cppn.generate_image_with_routing(params, image_id=image_id, img_size=img_size)
