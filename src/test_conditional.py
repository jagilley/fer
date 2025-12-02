"""Quick test to verify the conditional CPPN implementation works."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from cppn_conditional import ConditionalCPPN, FlattenConditionalCPPNParameters

# Test 1: Can we initialize the model?
print("Test 1: Initializing ConditionalCPPN...")
cppn = FlattenConditionalCPPNParameters(
    ConditionalCPPN(
        arch="12;cache:15,gaussian:4,identity:2,sin:1",
        n_images=3
    )
)
print(f"✓ Model initialized with {cppn.n_params} parameters")
print(f"  Expected: ~5478 base params + overhead for conditioning")

# Test 2: Can we initialize parameters?
print("\nTest 2: Initializing parameters...")
rng = jax.random.PRNGKey(0)
params = cppn.init(rng)
print(f"✓ Parameters initialized, shape: {params.shape}")

# Test 3: Can we generate images for each condition?
print("\nTest 3: Generating images for each image_id...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for img_id in range(3):
    print(f"  Generating image {img_id}...")
    img = cppn.generate_image(params, image_id=img_id, img_size=64)
    print(f"    ✓ Generated image with shape {img.shape}")

    axes[img_id].imshow(img)
    axes[img_id].set_title(f"Image ID: {img_id}")
    axes[img_id].axis('off')

plt.tight_layout()
plt.savefig("test_conditional_output.png", dpi=150, bbox_inches='tight')
print("\n✓ All tests passed!")
print("  Output saved to test_conditional_output.png")
print("\nNote: Images are random noise since we used random initialization.")
print("This just verifies the architecture works correctly.")
