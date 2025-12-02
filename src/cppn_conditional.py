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

cache = lambda x: x
identity = lambda x: x
cos = jnp.cos
sin = jnp.sin
tanh = jnp.tanh
sigmoid = lambda x: jax.nn.sigmoid(x) * 2. - 1.
gaussian = lambda x: jnp.exp(-x**2) * 2. - 1.
relu = jax.nn.relu
activation_fn_map = dict(cache=cache, identity=identity, cos=cos, sin=sin, tanh=tanh, sigmoid=sigmoid, gaussian=gaussian, relu=relu)

class ConditionalCPPN(nn.Module):
    """
    Conditional CPPN Flax Model that can generate multiple images based on an image_id.

    This extends the standard CPPN to accept an image_id conditioning input, allowing
    a single network to produce different images based on the conditioning.

    arch: str in the form "12;cache:15,gaussian:4,identity:2,sin:1"
    n_images: int number of images this CPPN can generate (determines size of one-hot encoding)
    inputs: str base inputs like "x,y,d,b" (image_id will be automatically appended as one-hot)
    init_scale: str "default" or float for initialization scale
    """
    arch: str = "12;cache:15,gaussian:4,identity:2,sin:1"
    n_images: int = 3  # number of images (skull, butterfly, apple)
    inputs: str = "x,y,d,b"
    init_scale: str = "default"

    @nn.compact
    def __call__(self, x):
        n_layers, activation_neurons = self.arch.split(";")
        n_layers = int(n_layers)

        activations = [i.split(":")[0] for i in activation_neurons.split(",")]
        d_hidden = [int(i.split(":")[-1]) for i in activation_neurons.split(",")]
        dh_cumsum = list(np.cumsum(d_hidden))

        features = [x]
        for i_layer in range(n_layers):
            if self.init_scale == "default":
                x = nn.Dense(sum(d_hidden), use_bias=False)(x)
            else:
                kernel_init = nn.initializers.variance_scaling(scale=float(self.init_scale), mode="fan_in", distribution="truncated_normal")
                x = nn.Dense(sum(d_hidden), use_bias=False, kernel_init=kernel_init)(x)

            x = jnp.split(x, dh_cumsum)
            x = [activation_fn_map[activation](xi) for xi, activation in zip(x, activations)]
            x = jnp.concatenate(x)

            features.append(x)
        x = nn.Dense(3, use_bias=False)(x)
        features.append(x)
        h, s, v = x
        return (h, s, v), features

    def generate_image(self, params, image_id, img_size=256, return_features=False):
        """
        Generate an image from the conditional CPPN.

        params: network parameters
        image_id: integer in [0, n_images) specifying which image to generate
        img_size: resolution of output image
        return_features: whether to return intermediate activations
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

class FlattenConditionalCPPNParameters():
    """
    Flatten the parameters of the conditional CPPN to a single vector.
    Similar to FlattenCPPNParameters but for the conditional version.
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
