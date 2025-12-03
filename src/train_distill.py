"""
Distillation training: Train a student model on teacher's simplex-sampled outputs.

Phase 1: Distillation - Train student to match teacher at sampled simplex points
Phase 2: Fine-tuning - Continue training on original ground truth images
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

from cppn_conditional import ConditionalCPPN, FlattenConditionalCPPNParameters
import util
import fer_metrics
import post_training_viz

parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0, help="random seed")
group.add_argument("--save_dir", type=str, required=True, help="path to save results")

group = parser.add_argument_group("teacher")
group.add_argument("--teacher_dir", type=str, required=True, help="directory with trained teacher model")

group = parser.add_argument_group("student")
group.add_argument("--student_arch", type=str, required=True, help="student architecture string")

group = parser.add_argument_group("data")
group.add_argument("--img_files", type=str, nargs='+', required=True, help="paths to ground truth images")

group = parser.add_argument_group("distillation")
group.add_argument("--distill_iters", type=int, default=50000, help="distillation iterations")
group.add_argument("--n_simplex_samples", type=int, default=15, help="samples per simplex (includes corners)")
group.add_argument("--distill_lr", type=float, default=3e-3, help="distillation learning rate")

group = parser.add_argument_group("finetuning")
group.add_argument("--finetune_iters", type=int, default=50000, help="fine-tuning iterations")
group.add_argument("--finetune_lr", type=float, default=1e-3, help="fine-tuning learning rate")

group = parser.add_argument_group("metrics")
group.add_argument("--track_metrics", action="store_true", help="track FER/UFR metrics")
group.add_argument("--metric_interval", type=int, default=2000, help="metric computation interval")


def generate_at_condition(cppn, params, condition, img_size=256):
    """
    Generate image at arbitrary simplex conditioning.
    Adapted from visualize_conditional_interpolations.py

    Args:
        cppn: FlattenConditionalCPPNParameters wrapper
        params: Flattened parameters
        condition: Array of shape (n_images,) with weights summing to 1
        img_size: Output resolution
    """
    from color import hsv2rgb

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


def sample_simplex_points(n_images, n_samples):
    """
    Sample points from the (n_images-1)-simplex using barycentric coordinates.
    Always includes the corners (one-hot vectors).

    Returns: list of condition arrays, each summing to 1
    """
    points = []

    # Always include corners (ground truth conditions)
    for i in range(n_images):
        corner = np.zeros(n_images)
        corner[i] = 1.0
        points.append(corner)

    # Sample interior points using triangular grid
    n_levels = int(np.sqrt(n_samples - n_images)) + 2

    if n_images == 3:
        # 2-simplex: triangular grid
        for i in range(n_levels):
            for j in range(n_levels - i):
                k = n_levels - 1 - i - j
                if i == 0 and j == 0:
                    continue  # skip corner
                if i == 0 and k == 0:
                    continue  # skip corner
                if j == 0 and k == 0:
                    continue  # skip corner
                condition = np.array([i, j, k], dtype=float) / (n_levels - 1)
                points.append(condition)

    return points


def main(args):
    print("="*60)
    print("DISTILLATION TRAINING")
    print("="*60)
    print(f"Teacher: {args.teacher_dir}")
    print(f"Student arch: {args.student_arch}")
    print(f"Distillation: {args.distill_iters} iters, {args.n_simplex_samples} simplex samples")
    print(f"Fine-tuning: {args.finetune_iters} iters on ground truth")
    print("="*60)

    # Load teacher
    print("\nLoading teacher model...")
    teacher_arch = util.load_pkl(args.teacher_dir, "arch")
    teacher_params = util.load_pkl(args.teacher_dir, "params")
    teacher_args = util.load_pkl(args.teacher_dir, "args")
    n_images = teacher_args.n_images

    teacher = FlattenConditionalCPPNParameters(
        ConditionalCPPN(arch=teacher_arch, n_images=n_images)
    )
    print(f"Teacher: {teacher_arch} ({teacher.n_params} params)")

    # Initialize student
    print("\nInitializing student model...")
    student = FlattenConditionalCPPNParameters(
        ConditionalCPPN(arch=args.student_arch, n_images=n_images)
    )
    print(f"Student: {args.student_arch} ({student.n_params} params)")

    rng = jax.random.PRNGKey(args.seed)
    student_params = student.init(rng)

    # Load ground truth images for fine-tuning
    target_imgs = []
    for img_file in args.img_files:
        img = jnp.array(plt.imread(img_file)[:, :, :3])
        target_imgs.append(img)
    target_imgs = jnp.stack(target_imgs)
    print(f"Loaded {len(target_imgs)} ground truth images")

    # Sample simplex points for distillation
    simplex_points = sample_simplex_points(n_images, args.n_simplex_samples)
    print(f"Sampled {len(simplex_points)} simplex points for distillation")

    # Pre-generate teacher outputs at all simplex points
    print("\nGenerating teacher outputs...")
    teacher_outputs = {}
    for i, condition in enumerate(tqdm(simplex_points)):
        key = tuple(condition)
        teacher_outputs[key] = generate_at_condition(teacher, teacher_params, condition, img_size=256)
    print(f"Generated {len(teacher_outputs)} teacher images")

    # Convert to arrays for efficient training
    conditions_array = jnp.array(simplex_points)
    teacher_imgs_array = jnp.stack([teacher_outputs[tuple(c)] for c in simplex_points])

    # Metric tracking
    metric_history = {
        "steps": [], "phase": [],
        "feature_similarity": [], "neuron_specialization": [], "interpolation_smoothness": []
    }
    losses = []

    # ===================
    # PHASE 1: DISTILLATION
    # ===================
    print("\n" + "="*60)
    print("PHASE 1: DISTILLATION")
    print("="*60)

    def distill_loss_fn(params, teacher_imgs, conditions):
        """MSE loss between student and teacher outputs at simplex points."""
        total_loss = 0.0
        n_samples = len(conditions)
        for i in range(n_samples):
            student_img = generate_at_condition(student, params, conditions[i], img_size=256)
            teacher_img = teacher_imgs[i]
            total_loss += jnp.mean((student_img - teacher_img)**2)
        return total_loss / n_samples

    # For efficiency, use a batched approach with random sampling each step
    @jax.jit
    def distill_step(state, rng):
        # Sample a subset of simplex points each step for efficiency
        n_batch = min(5, len(simplex_points))
        indices = jax.random.choice(rng, len(simplex_points), shape=(n_batch,), replace=False)

        batch_conditions = conditions_array[indices]
        batch_teachers = teacher_imgs_array[indices]

        def batch_loss(params):
            total = 0.0
            for i in range(n_batch):
                pred = generate_at_condition(student, params, batch_conditions[i], img_size=256)
                total += jnp.mean((pred - batch_teachers[i])**2)
            return total / n_batch

        loss, grad = jax.value_and_grad(batch_loss)(state.params)
        grad = grad / (jnp.linalg.norm(grad) + 1e-8)
        state = state.apply_gradients(grads=grad)
        return state, loss

    tx = optax.adam(learning_rate=args.distill_lr)
    state = TrainState.create(apply_fn=None, params=student_params, tx=tx)

    pbar = tqdm(range(args.distill_iters // 100))
    for i_outer in pbar:
        batch_losses = []
        for i_inner in range(100):
            rng, step_rng = jax.random.split(rng)
            state, loss = distill_step(state, step_rng)
            batch_losses.append(float(loss))
        losses.extend(batch_losses)
        pbar.set_postfix(loss=f"{np.mean(batch_losses):.6f}", phase="distill")

        # Track metrics
        current_step = (i_outer + 1) * 100
        if args.track_metrics and current_step % args.metric_interval == 0:
            metrics = fer_metrics.compute_all_metrics(student, state.params, n_images)
            metric_history["steps"].append(current_step)
            metric_history["phase"].append("distill")
            metric_history["feature_similarity"].append(metrics["feature_similarity"])
            metric_history["neuron_specialization"].append(metrics["neuron_specialization"])
            metric_history["interpolation_smoothness"].append(metrics["interpolation_smoothness"])

    distill_final_loss = losses[-1]
    print(f"Distillation complete. Final loss: {distill_final_loss:.6f}")

    # ===================
    # PHASE 2: FINE-TUNING
    # ===================
    print("\n" + "="*60)
    print("PHASE 2: FINE-TUNING ON GROUND TRUTH")
    print("="*60)

    def finetune_loss_fn(params, target_imgs):
        """MSE loss on ground truth images only."""
        total_loss = 0.0
        for img_id in range(n_images):
            pred_img = student.generate_image(params, image_id=img_id, img_size=256)
            target_img = target_imgs[img_id]
            total_loss += jnp.mean((pred_img - target_img)**2)
        return total_loss / n_images

    @jax.jit
    def finetune_step(state, _):
        loss, grad = jax.value_and_grad(finetune_loss_fn)(state.params, target_imgs)
        grad = grad / (jnp.linalg.norm(grad) + 1e-8)
        state = state.apply_gradients(grads=grad)
        return state, loss

    # Reset optimizer for fine-tuning with potentially different LR
    tx = optax.adam(learning_rate=args.finetune_lr)
    state = TrainState.create(apply_fn=None, params=state.params, tx=tx)

    pbar = tqdm(range(args.finetune_iters // 100))
    for i_iter in pbar:
        state, loss = jax.lax.scan(finetune_step, state, None, length=100)
        losses.extend([float(l) for l in loss])
        pbar.set_postfix(loss=f"{loss.mean():.6f}", phase="finetune")

        # Track metrics
        if args.track_metrics:
            current_step = args.distill_iters + (i_iter + 1) * 100
            if current_step % args.metric_interval == 0:
                metrics = fer_metrics.compute_all_metrics(student, state.params, n_images)
                metric_history["steps"].append(current_step)
                metric_history["phase"].append("finetune")
                metric_history["feature_similarity"].append(metrics["feature_similarity"])
                metric_history["neuron_specialization"].append(metrics["neuron_specialization"])
                metric_history["interpolation_smoothness"].append(metrics["interpolation_smoothness"])

    finetune_final_loss = losses[-1]
    print(f"Fine-tuning complete. Final loss: {finetune_final_loss:.6f}")

    # ===================
    # SAVE RESULTS
    # ===================
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    os.makedirs(args.save_dir, exist_ok=True)

    final_params = state.params
    util.save_pkl(args.save_dir, "args", args)
    util.save_pkl(args.save_dir, "arch", args.student_arch)
    util.save_pkl(args.save_dir, "params", final_params)
    util.save_pkl(args.save_dir, "losses", np.array(losses))

    # Save generated images
    for img_id in range(n_images):
        img = student.generate_image(final_params, image_id=img_id, img_size=256)
        plt.imsave(f"{args.save_dir}/img_{img_id}.png", np.array(img))

    # Save comparison grid
    fig, axes = plt.subplots(2, n_images, figsize=(5*n_images, 10))
    for img_id in range(n_images):
        # Ground truth
        axes[0, img_id].imshow(target_imgs[img_id])
        axes[0, img_id].set_title(f"Ground Truth {img_id}")
        axes[0, img_id].axis('off')

        # Student output
        student_img = student.generate_image(final_params, image_id=img_id, img_size=256)
        axes[1, img_id].imshow(np.array(student_img))
        axes[1, img_id].set_title(f"Student {img_id}")
        axes[1, img_id].axis('off')

    plt.suptitle(f"Distillation: {teacher_arch} → {args.student_arch}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{args.save_dir}/all_images.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save metrics if tracked
    if args.track_metrics and len(metric_history["steps"]) > 0:
        util.save_pkl(args.save_dir, "fer_metrics", metric_history)

        # Plot metrics with phase indicator
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss curve
        axes[0, 0].plot(losses, alpha=0.3, color='darkred')
        if len(losses) > 1000:
            losses_smooth = np.convolve(losses, np.ones(1000)/1000, mode='valid')
            axes[0, 0].plot(losses_smooth, color='darkred', linewidth=2)
        axes[0, 0].axvline(x=args.distill_iters, color='blue', linestyle='--', label='Finetune start')
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        steps = metric_history["steps"]

        # Feature similarity
        axes[0, 1].plot(steps, metric_history["feature_similarity"], marker='o', color='blue')
        axes[0, 1].axvline(x=args.distill_iters, color='gray', linestyle='--')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Feature Correlation')
        axes[0, 1].set_title('Feature Similarity (↑ = UFR)')
        axes[0, 1].grid(True, alpha=0.3)

        # Neuron specialization
        axes[1, 0].plot(steps, metric_history["neuron_specialization"], marker='o', color='orange')
        axes[1, 0].axvline(x=args.distill_iters, color='gray', linestyle='--')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Coefficient of Variation')
        axes[1, 0].set_title('Neuron Specialization (↓ = UFR)')
        axes[1, 0].grid(True, alpha=0.3)

        # Interpolation smoothness
        axes[1, 1].plot(steps, metric_history["interpolation_smoothness"], marker='o', color='purple')
        axes[1, 1].axvline(x=args.distill_iters, color='gray', linestyle='--')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Smoothness Score')
        axes[1, 1].set_title('Interpolation Smoothness (↑ = UFR)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f'FER/UFR Metrics: Distill ({args.distill_iters}) → Finetune ({args.finetune_iters})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{args.save_dir}/fer_metrics.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Run post-training visualizations
    post_training_viz.run_all_visualizations(student, final_params, args.save_dir, n_images=n_images)

    print(f"\nResults saved to {args.save_dir}")
    print("="*60)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
