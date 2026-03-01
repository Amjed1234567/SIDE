import torch
import numpy as np
import matplotlib.pyplot as plt

trained_model = torch.load("dim_2_3.pt", map_location="cpu")

w = trained_model["w"]
v = trained_model["v"]

w_np = w.numpy()
v_np = v.numpy()

plt.figure(figsize=(8, 6))
POINT_SIZE    = 30
DRUG_COLOR    = "blue" 
DRUG_MARKER   = "o"
SE_COLOR = "orange"
SE_MARKER = "^"

# Drugs (w)
plt.scatter(
    w_np[:, 0], w_np[:, 1],
    s=POINT_SIZE,
    c=DRUG_COLOR,
    marker=DRUG_MARKER,
    label=f"Drugs (n={w_np.shape[0]})",
    edgecolor="black", linewidth=0.5, alpha=0.8
)

# Side‑effects (v)
plt.scatter(
    v_np[:, 0], v_np[:, 1],
    s=POINT_SIZE,
    c=SE_COLOR,
    marker=SE_MARKER,
    label=f"Side-effects (n={v_np.shape[0]})",
    edgecolor="black", linewidth=0.5, alpha=0.8
)

plt.title("Latent-space embedding (2-D) - drugs vs. side-effects")
plt.xlabel("Latent dimension_1")
plt.ylabel("Latent dimension_2")
plt.legend(loc="best")
plt.grid(True, linestyle="--", alpha=0.4)

# Ensure equal aspect ratio so distances are not distorted
plt.axis("equal")

plt.tight_layout()
plt.savefig("scatter_vw.png", dpi=300, bbox_inches="tight")
print(f"Scatter plot saved ")