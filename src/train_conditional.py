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

from cppn_conditional import ConditionalCPPN, FlattenConditionalCPPNParameters
import util

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0, help="the random seed")
group.add_argument("--save_dir", type=str, default=None, help="path to save results to")

group = parser.add_argument_group("model")
group.add_argument("--arch", type=str, default="12;cache:15,gaussian:4,identity:2,sin:1", help="architecture")
group.add_argument("--n_images", type=int, default=3, help="number of images to condition on")

group = parser.add_argument_group("data")
group.add_argument("--img_files", type=str, nargs='+', required=True, help="paths to image files (in order: 0, 1, 2, ...)")

group = parser.add_argument_group("optimization")
group.add_argument("--n_iters", type=int, default=100000, help="number of iterations")
group.add_argument("--lr", type=float, default=3e-3, help="learning rate")
group.add_argument("--init_scale", type=str, default="default", help="initialization scale")
group.add_argument("--mode", type=str, default="direct", choices=["direct", "distill"],
                   help="training mode: 'direct' trains from scratch, 'distill' trains from teacher outputs")
group.add_argument("--teacher_dir", type=str, default=None, help="directory containing teacher model (required for distill mode)")

def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)

    # Validation
    if args.mode == "distill" and args.teacher_dir is None:
        raise ValueError("--teacher_dir is required when mode is 'distill'")
    if len(args.img_files) != args.n_images:
        raise ValueError(f"Number of img_files ({len(args.img_files)}) must match n_images ({args.n_images})")

    return args

def main(args):
    """
    Train a conditional CPPN on multiple images.

    Two modes:
    - direct: Train from scratch to match target images directly
    - distill: Train to match a teacher model's outputs (for distillation experiments)
    """
    print(args)

    # Load all target images
    target_imgs = []
    for img_file in args.img_files:
        img = jnp.array(plt.imread(img_file)[:, :, :3])
        target_imgs.append(img)
    target_imgs = jnp.stack(target_imgs)  # shape: (n_images, H, W, 3)

    print(f"Loaded {len(target_imgs)} target images with shape {target_imgs[0].shape}")

    # Initialize conditional CPPN
    cppn = FlattenConditionalCPPNParameters(
        ConditionalCPPN(arch=args.arch, n_images=args.n_images, init_scale=args.init_scale)
    )
    print(f"Conditional CPPN with {cppn.n_params} parameters")

    rng = jax.random.PRNGKey(args.seed)
    params = cppn.init(rng)

    def loss_fn(params, target_imgs):
        """
        Compute MSE loss across all images.
        """
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
    pbar = tqdm(range(args.n_iters//100))
    for i_iter in pbar:
        state, loss = jax.lax.scan(train_step, state, None, length=100)
        losses.append(loss)
        pbar.set_postfix(loss=loss.mean().item())

    losses = np.array(jnp.concatenate(losses))
    params = state.params

    # Generate final images for all conditions
    final_imgs = []
    for img_id in range(args.n_images):
        img = cppn.generate_image(params, image_id=img_id, img_size=256)
        final_imgs.append(np.array(img))

    # Save results
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        util.save_pkl(args.save_dir, "args", args)
        util.save_pkl(args.save_dir, "arch", args.arch)
        util.save_pkl(args.save_dir, "params", params)
        util.save_pkl(args.save_dir, "losses", losses)

        # Save individual images
        for img_id, img in enumerate(final_imgs):
            plt.imsave(f"{args.save_dir}/img_{img_id}.png", img)

        # Save a grid of all images
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

        print(f"Results saved to {args.save_dir}")

if __name__ == '__main__':
    main(parse_args())
