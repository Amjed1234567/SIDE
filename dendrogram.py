import torch
import numpy as np
import matplotlib.pyplot as plt
# From https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
# and https://www.datasciencebase.com/unsupervised-ml/algorithms/agglomerative-hierarchical-clustering/pytorch-example/
from scipy.cluster.hierarchy import linkage, dendrogram


trained_model = torch.load("dim_2_3.pt", map_location="cpu")

w = trained_model["w"]
v = trained_model["v"]

w_np = w.numpy()
v_np = v.numpy()

Z = torch.cat([w, v], dim=0).numpy()   # shape (n_drugs + n_ses, latent_dim)

FIGSIZE = (20, 10)
DENDRO_ORIENTATION = "top" 
LABEL_FONTSIZE = 8
TITLE = "Ward Hierarchical Clustering of Latent Embeddings (w + v)"

linkage_matrix = linkage(Z, method="ward", metric="euclidean")
plt.figure(figsize=FIGSIZE)

dendro = dendrogram(
    linkage_matrix,
    orientation=DENDRO_ORIENTATION,
    leaf_rotation=90,          # make leaf labels vertical
    leaf_font_size=LABEL_FONTSIZE,
    no_labels=True,
    color_threshold=None,      # color all branches the same 
    above_threshold_color="grey"
)

plt.title(TITLE, fontsize=14)
plt.xlabel("Samples", fontsize=12)
plt.ylabel("Distance", fontsize=12)

plt.tight_layout()
plt.savefig("dendrogram.png", dpi=300, bbox_inches="tight")
print(f"Dendrogram saved.")
