"""
Training script for MoE Conditional CPPN.

Tests the hypothesis that MoE architectures achieve better (more UFR-like)
representations than dense models of equivalent parameter count.
"""

import os
from functools import partial
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from cppn_moe import MoEConditionalCPPN, FlattenMoECPPNParameters
import util
import fer_metrics
import post_training_viz

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0, help="the random seed")
group.add_argument("--save_dir", type=str, default=None, help="path to save results to")

group = parser.add_argument_group("model")
group.add_argument("--expert_arch", type=str, default="6;cache:8,gaussian:2,identity:1,sin:1",
                   help="architecture for each expert (default: tiny)")
group.add_argument("--n_experts", type=int, default=6, help="number of experts")
group.add_argument("--router_hidden", type=int, default=48, help="router hidden dimension")
group.add_argument("--n_images", type=int, default=3, help="number of images to condition on")

group = parser.add_argument_group("data")
group.add_argument("--img_files", type=str, nargs='+', required=True, help="paths to image files")

group = parser.add_argument_group("optimization")
group.add_argument("--n_iters", type=int, default=100000, help="number of iterations")
group.add_argument("--lr", type=float, default=3e-3, help="learning rate")

group = parser.add_argument_group("metrics")
group.add_argument("--track_metrics", action="store_true", help="track FER/UFR metrics during training")
group.add_argument("--metric_interval", type=int, default=2000, help="compute metrics every N iterations")


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)

    if len(args.img_files) != args.n_images:
        raise ValueError(f"Number of img_files ({len(args.img_files)}) must match n_images ({args.n_images})")

    return args


def visualize_routing(cppn, params, save_dir, n_images=3, img_size=256):
    """
    Visualize the routing weights for each expert across space.
    """
    # Generate coordinate grid
    x = y = jnp.linspace(-1, 1, img_size)
    grid_x, grid_y = jnp.meshgrid(x, y, indexing='ij')
    d = jnp.sqrt(grid_x**2 + grid_y**2) * 1.4
    b = jnp.ones_like(grid_x)
    spatial_coords = jnp.stack([grid_x, grid_y, d, b], axis=-1)

    # Get unflattened params
    params_dict = cppn.param_reshaper.reshape_single(params)

    # Apply router to get weights at each spatial location
    router_params = params_dict['params']['router']
    hidden = jnp.tanh(jax.vmap(jax.vmap(
        lambda sc: jnp.dot(sc, router_params['Dense_0']['kernel'])
    ))(spatial_coords))
    logits = jax.vmap(jax.vmap(
        lambda h: jnp.dot(h, router_params['Dense_1']['kernel'])
    ))(hidden)
    routing_weights = jax.nn.softmax(logits, axis=-1)  # (H, W, n_experts)

    n_experts = routing_weights.shape[-1]

    # Plot routing weights for each expert
    fig, axes = plt.subplots(2, (n_experts + 1) // 2, figsize=(4 * ((n_experts + 1) // 2), 8))
    axes = axes.flatten()

    for i in range(n_experts):
        ax = axes[i]
        im = ax.imshow(routing_weights[:, :, i].T, cmap='viridis', vmin=0, vmax=1, origin='lower')
        ax.set_title(f'Expert {i} weight')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Hide unused axes
    for i in range(n_experts, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Routing Weights by Spatial Position\n(Image-agnostic: same for all images)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/routing_weights.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Also show dominant expert map
    dominant_expert = jnp.argmax(routing_weights, axis=-1)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(dominant_expert.T, cmap='tab10', vmin=0, vmax=n_experts-1, origin='lower')
    ax.set_title('Dominant Expert by Position')
    ax.axis('off')
    plt.colorbar(im, ax=ax, ticks=range(n_experts), label='Expert ID')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/dominant_expert.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Routing visualizations saved to {save_dir}/")


def main(args):
    """
    Train a MoE conditional CPPN on multiple images.
    """
    print(args)

    # Load all target images
    target_imgs = []
    for img_file in args.img_files:
        img = jnp.array(plt.imread(img_file)[:, :, :3])
        target_imgs.append(img)
    target_imgs = jnp.stack(target_imgs)

    print(f"Loaded {len(target_imgs)} target images with shape {target_imgs[0].shape}")

    # Initialize MoE CPPN
    cppn = FlattenMoECPPNParameters(
        MoEConditionalCPPN(
            expert_arch=args.expert_arch,
            n_experts=args.n_experts,
            n_images=args.n_images,
            router_hidden=args.router_hidden
        )
    )
    print(f"MoE CPPN with {cppn.n_params} parameters ({args.n_experts} experts)")

    rng = jax.random.PRNGKey(args.seed)
    params = cppn.init(rng)

    def loss_fn(params, target_imgs):
        """Compute MSE loss across all images."""
        total_loss = 0.0
        for img_id in range(args.n_images):
            pred_img = cppn.generate_image(params, image_id=img_id, img_size=256)
            target_img = target_imgs[img_id]
            total_loss += jnp.mean((pred_img - target_img)**2)
        return total_loss / args.n_images

    @jax.jit
    def train_step(state, _):
        loss, grad = jax.value_and_grad(loss_fn)(state.params, target_imgs)
        grad = grad / jnp.linalg.norm(grad)
        state = state.apply_gradients(grads=grad)
        return state, loss

    tx = optax.adam(learning_rate=args.lr)
    state = TrainState.create(apply_fn=None, params=params, tx=tx)

    # Training loop
    losses = []
    metric_history = {
        "steps": [],
        "feature_similarity": [],
        "neuron_specialization": [],
        "interpolation_smoothness": [],
        "spatial_roughness": [],
        "max_roughness": [],
        "effective_dim_ratio": []
    }

    pbar = tqdm(range(args.n_iters // 100))
    for i_iter in pbar:
        state, loss = jax.lax.scan(train_step, state, None, length=100)
        losses.append(loss)
        pbar.set_postfix(loss=loss.mean().item())

        # Track metrics if enabled
        if args.track_metrics:
            current_step = (i_iter + 1) * 100
            if current_step % args.metric_interval == 0 or current_step == args.n_iters:
                metrics = fer_metrics.compute_all_metrics(cppn, state.params, args.n_images)
                metric_history["steps"].append(current_step)
                metric_history["feature_similarity"].append(metrics["feature_similarity"])
                metric_history["neuron_specialization"].append(metrics["neuron_specialization"])
                metric_history["interpolation_smoothness"].append(metrics["interpolation_smoothness"])
                metric_history["spatial_roughness"].append(metrics["spatial_roughness"])
                metric_history["max_roughness"].append(metrics["max_roughness"])
                metric_history["effective_dim_ratio"].append(metrics["effective_dim_ratio"])
                fer_metrics.print_metrics(metrics, step=current_step)

    losses = np.array(jnp.concatenate(losses))
    params = state.params

    # Generate final images
    final_imgs = []
    for img_id in range(args.n_images):
        img = cppn.generate_image(params, image_id=img_id, img_size=256)
        final_imgs.append(np.array(img))

    # Save results
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        util.save_pkl(args.save_dir, "args", args)
        util.save_pkl(args.save_dir, "arch", f"MoE:{args.expert_arch}x{args.n_experts}")
        util.save_pkl(args.save_dir, "params", params)
        util.save_pkl(args.save_dir, "losses", losses)

        # Save individual images
        for img_id, img in enumerate(final_imgs):
            plt.imsave(f"{args.save_dir}/img_{img_id}.png", img)

        # Save grid of all images
        fig, axes = plt.subplots(1, args.n_images, figsize=(5*args.n_images, 5))
        if args.n_images == 1:
            axes = [axes]
        for img_id, (ax, img) in enumerate(zip(axes, final_imgs)):
            ax.imshow(img)
            ax.set_title(f"Image {img_id}")
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"{args.save_dir}/all_images.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Visualize routing weights
        visualize_routing(cppn, params, args.save_dir, n_images=args.n_images)

        # Save and plot metrics if tracked
        if args.track_metrics and len(metric_history["steps"]) > 0:
            util.save_pkl(args.save_dir, "fer_metrics", metric_history)

            # Plot metrics (extended for MoE)
            fig, axes = plt.subplots(3, 2, figsize=(14, 15))

            # Loss curve
            axes[0, 0].plot(losses, alpha=0.3, color='darkred')
            losses_smooth = np.convolve(losses, np.ones(1000)/1000, mode='valid')
            axes[0, 0].plot(losses_smooth, color='darkred', linewidth=2)
            axes[0, 0].set_yscale('log')
            axes[0, 0].set_xlabel('Training Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].grid(True, alpha=0.3)

            # Feature similarity
            axes[0, 1].plot(metric_history["steps"], metric_history["feature_similarity"],
                          marker='o', color='blue', linewidth=2)
            axes[0, 1].axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='UFR threshold')
            axes[0, 1].axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='FER threshold')
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('Feature Correlation')
            axes[0, 1].set_title('Feature Similarity')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(0, 1)

            # Neuron specialization
            axes[1, 0].plot(metric_history["steps"], metric_history["neuron_specialization"],
                          marker='o', color='orange', linewidth=2)
            axes[1, 0].axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='UFR threshold')
            axes[1, 0].axhline(y=0.4, color='red', linestyle='--', alpha=0.5, label='FER threshold')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Coefficient of Variation')
            axes[1, 0].set_title('Neuron Specialization')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Interpolation smoothness
            axes[1, 1].plot(metric_history["steps"], metric_history["interpolation_smoothness"],
                          marker='o', color='purple', linewidth=2)
            axes[1, 1].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='UFR threshold')
            axes[1, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='FER threshold')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Smoothness Score')
            axes[1, 1].set_title('Interpolation Smoothness')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(0, 1)

            # Spatial roughness (KEY METRIC)
            axes[2, 0].plot(metric_history["steps"], metric_history["spatial_roughness"],
                          marker='o', color='crimson', linewidth=2, label='Avg Roughness')
            axes[2, 0].plot(metric_history["steps"], metric_history["max_roughness"],
                          marker='s', color='darkred', linewidth=2, alpha=0.7, label='Max Roughness')
            axes[2, 0].axhline(y=0.02, color='green', linestyle='--', alpha=0.5, label='UFR avg (~0.02)')
            axes[2, 0].axhline(y=0.07, color='red', linestyle='--', alpha=0.5, label='FER avg (~0.07)')
            axes[2, 0].set_xlabel('Training Step')
            axes[2, 0].set_ylabel('Roughness')
            axes[2, 0].set_title('Spatial Roughness (KEY: lower = more UFR)')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)

            # Effective dimensionality (KEY METRIC)
            axes[2, 1].plot(metric_history["steps"], metric_history["effective_dim_ratio"],
                          marker='o', color='teal', linewidth=2)
            axes[2, 1].axhline(y=0.4, color='green', linestyle='--', alpha=0.5, label='UFR threshold (~0.3-0.4)')
            axes[2, 1].axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='FER threshold (~0.8)')
            axes[2, 1].set_xlabel('Training Step')
            axes[2, 1].set_ylabel('Max Effective Rank / Neurons')
            axes[2, 1].set_title('Effective Dim Ratio (KEY: lower = more UFR)')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
            axes[2, 1].set_ylim(0, 1)

            plt.suptitle(f'FER/UFR Metrics - MoE ({args.n_experts} experts, {cppn.n_params} params)',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{args.save_dir}/fer_metrics.png", dpi=150, bbox_inches='tight')
            plt.close()

            print(f"FER/UFR metrics saved to {args.save_dir}/fer_metrics.png")

        # Run post-training visualizations
        post_training_viz.run_all_visualizations(cppn, params, args.save_dir, n_images=args.n_images)

        print(f"Results saved to {args.save_dir}")


if __name__ == '__main__':
    main(parse_args())
