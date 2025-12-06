"""
Plot loss vs FER metrics (spatial roughness, effective dim ratio) to analyze
the tradeoff between reconstruction quality and representation quality.

Shows that distillation can beat the Pareto frontier established by direct training.
"""

import numpy as np
import matplotlib.pyplot as plt

# Data from experiments
EXPERIMENTS = {
    'tiny_direct': {'params': 840, 'loss': 0.0130, 'roughness': 0.042, 'eff_dim': 0.793},
    'tiny_long': {'params': 840, 'loss': 0.0129, 'roughness': 0.042, 'eff_dim': 0.829},
    'intermediate': {'params': 2482, 'loss': 0.0051, 'roughness': 0.055, 'eff_dim': 0.830},
    'small_direct': {'params': 5544, 'loss': 0.0020, 'roughness': 0.063, 'eff_dim': 0.875},
    'small_distill': {'params': 5544, 'loss': 0.0026, 'roughness': 0.055, 'eff_dim': 0.830},
}

# Label positions (x_offset, y_offset) for each experiment and metric
LABEL_OFFSETS = {
    'roughness': {
        'tiny_direct': (0.0003, 0.001),
        'intermediate': (0.0003, 0.001),
        'small_direct': (0.0003, 0.001),
        'small_distill': (0.0003, 0.001),
        'tiny_long': (-0.004, 0.001),
    },
    'eff_dim': {
        'tiny_direct': (0.0003, 0.005),
        'intermediate': (0.0003, 0.005),
        'small_direct': (0.0003, 0.005),
        'small_distill': (0.0003, 0.005),
        'tiny_long': (-0.004, 0.005),
    }
}


def plot_loss_vs_fer_metrics(save_path='../experiments/loss_vs_fer_metrics.png'):
    """Create scatter plot of loss vs FER metrics with trendline."""

    # Separate direct-trained vs distilled
    direct = ['tiny_direct', 'intermediate', 'small_direct']
    distilled = ['small_distill']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, (metric, label, target) in enumerate([
        ('roughness', 'Spatial Roughness', 0.02),
        ('eff_dim', 'Effective Dim Ratio', 0.45)
    ]):
        ax = axes[ax_idx]

        # Plot direct-trained models
        direct_losses = [EXPERIMENTS[k]['loss'] for k in direct]
        direct_metrics = [EXPERIMENTS[k][metric] for k in direct]
        ax.scatter(direct_losses, direct_metrics, s=100, c='blue', label='Direct training', zorder=3)

        # Add labels for direct
        for name in direct:
            d = EXPERIMENTS[name]
            offset = LABEL_OFFSETS[metric][name]
            display_name = name.replace('_direct', '')
            ax.annotate(display_name, (d['loss'], d[metric]),
                       xytext=(d['loss'] + offset[0], d[metric] + offset[1]), fontsize=9)

        # Fit trendline to direct-trained models only
        z = np.polyfit(direct_losses, direct_metrics, 1)
        p = np.poly1d(z)
        x_line = np.linspace(0.001, 0.015, 100)
        ax.plot(x_line, p(x_line), 'b--', alpha=0.5, label=f'Direct trend (slope={z[0]:.1f})')

        # Plot distilled model
        for name in distilled:
            d = EXPERIMENTS[name]
            ax.scatter(d['loss'], d[metric], s=150, c='red', marker='*', label='Distilled', zorder=4)
            offset = LABEL_OFFSETS[metric][name]
            ax.annotate('small_distill', (d['loss'], d[metric]),
                       xytext=(d['loss'] + offset[0], d[metric] + offset[1]), fontsize=9, color='red')

            # Show where distilled would be expected on trendline
            expected = p(d['loss'])
            ax.plot([d['loss'], d['loss']], [d[metric], expected], 'r:', alpha=0.7)
            improvement = expected - d[metric]
            delta_y_pos = (d[metric] + expected) / 2
            ax.annotate(f'Δ={improvement:.3f}', (d['loss'] + 0.0002, delta_y_pos),
                       fontsize=8, color='red')

        # Plot tiny_long to show extended training effect
        d = EXPERIMENTS['tiny_long']
        ax.scatter(d['loss'], d[metric], s=100, c='orange', marker='s', label='Extended training', zorder=3)
        offset = LABEL_OFFSETS[metric]['tiny_long']
        ax.annotate('tiny_long', (d['loss'], d[metric]),
                   xytext=(d['loss'] + offset[0], d[metric] + offset[1]), fontsize=9, color='orange')

        # UFR target line
        ax.axhline(y=target, color='green', linestyle='--', alpha=0.5, label=f'UFR target ({target})')

        ax.set_xlabel('Final Loss (MSE)', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'Loss vs {label}', fontsize=12)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.015)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {save_path}")

    return fig, axes


def print_distillation_analysis():
    """Print quantitative analysis of distillation effect."""

    print("\n=== Distillation Effect ===")
    d = EXPERIMENTS['small_distill']
    sd = EXPERIMENTS['small_direct']

    print(f"\nsmall_direct:  loss={sd['loss']:.4f}, roughness={sd['roughness']:.3f}, eff_dim={sd['eff_dim']:.3f}")
    print(f"small_distill: loss={d['loss']:.4f}, roughness={d['roughness']:.3f}, eff_dim={d['eff_dim']:.3f}")
    print(f"\nLoss increase: {(d['loss'] - sd['loss'])/sd['loss']*100:.1f}%")
    print(f"Roughness decrease: {(sd['roughness'] - d['roughness'])/sd['roughness']*100:.1f}%")
    print(f"Eff dim decrease: {(sd['eff_dim'] - d['eff_dim'])/sd['eff_dim']*100:.1f}%")

    # What loss would we expect for small_distill's roughness on the direct trendline?
    direct = ['tiny_direct', 'intermediate', 'small_direct']
    direct_losses = [EXPERIMENTS[k]['loss'] for k in direct]
    direct_roughness = [EXPERIMENTS[k]['roughness'] for k in direct]

    z_rough = np.polyfit(direct_losses, direct_roughness, 1)
    # Solve for loss: roughness = z[0]*loss + z[1] => loss = (roughness - z[1])/z[0]
    expected_loss_for_roughness = (d['roughness'] - z_rough[1]) / z_rough[0]

    print(f"\nExpected loss for roughness={d['roughness']:.3f} on direct trendline: {expected_loss_for_roughness:.4f}")
    print(f"Actual loss: {d['loss']:.4f}")
    print(f"Distillation achieves {(1 - d['loss']/expected_loss_for_roughness)*100:.1f}% better loss than trendline")


if __name__ == '__main__':
    plot_loss_vs_fer_metrics()
    print_distillation_analysis()
