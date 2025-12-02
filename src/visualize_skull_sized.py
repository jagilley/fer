"""Visualize the skull-sized reconstructions of butterfly and apple"""

import numpy as np
import matplotlib.pyplot as plt

from cppn import CPPN, FlattenCPPNParameters
import util

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

genomes = ["skull", "butterfly", "apple"]
sources = ["picbreeder", "sgd", "sgd_skull_sized"]

for col, genome in enumerate(genomes):
    for row, (source, label) in enumerate([
        ("picbreeder", "Picbreeder Original"),
        ("sgd", f"SGD Full-Size\n({genome} arch)"),
        ("sgd_skull_sized", "SGD Skull-Sized")
    ]):
        # Handle different naming conventions
        if source == "sgd_skull_sized":
            save_dir = f"../data/sgd_{genome}_skull_sized"
        else:
            save_dir = f"../data/{source}_{genome}"

        # Skip if directory doesn't exist
        import os
        if not os.path.exists(save_dir):
            axes[row, col].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=20)
            axes[row, col].axis('off')
            continue

        arch = util.load_pkl(save_dir, "arch")
        params = util.load_pkl(save_dir, "params")
        cppn = FlattenCPPNParameters(CPPN(arch))
        img = np.array(cppn.generate_image(params, img_size=256))

        axes[row, col].imshow(img)
        axes[row, col].set_title(f"{genome.capitalize()}\n{label}", fontsize=10)
        axes[row, col].axis('off')

        # Add border
        for spine in axes[row, col].spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

plt.suptitle("Compression Test: Skull-Sized Architecture (5,478 params)", fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig("skull_sized_compression_test.png", dpi=150, bbox_inches='tight')
print("Saved to skull_sized_compression_test.png")
