import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# From https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
# and https://www.datasciencebase.com/unsupervised-ml/algorithms/agglomerative-hierarchical-clustering/pytorch-example/
from scipy.cluster.hierarchy import linkage, dendrogram

data_file = "meddra_all_se.tsv.xlsx" 
drug_column_index = 0 
SE_column_index = 4 # Side effects.

df = pd.read_excel(data_file, header=None)
# Convert Dataframe to 2D list.
data_list_2D = df.values.tolist()

# Convert Dataframe columns to lists.
drug_list = df[drug_column_index].tolist()   
SE_list = df[SE_column_index].tolist() 

# Convert lists to sets (remove duplicates).
drug_set = set(drug_list)
SE_set = set(SE_list) 
# Usefull constants.
total_no_of_drugs = len(drug_set)
total_no_of_SE = len(SE_set)   

# Look-up dictionaries. Key is a unique number, and value is drug/SE.
number_drug_dict = {index: value for index, value in enumerate(drug_set)}
number_SE_dict = {index: value for index, value in enumerate(SE_set)}

# Look-up dictionaries. Key is drug/SE, and value is the same unique number as above.
# Keys and values are swapped compared to the above dictionaries. 
drug_number_dict = dict([(value, key) for key, value in number_drug_dict.items()])
SE_number_dict = dict([(value, key) for key, value in number_SE_dict.items()])

# Create Y matrix.
Y = torch.zeros([total_no_of_drugs, total_no_of_SE], dtype=torch.int32)
for row in data_list_2D:
    i = drug_number_dict[row[drug_column_index]]
    j = SE_number_dict[row[SE_column_index]]
    Y[i][j] = 1        

Y_np = Y.numpy()   # (n_drugs, n_ses) not ordered. 

# From 
def leaf_order_from_linkage(latent):
    """
    Returns a list of indices that correspond to the leaf order
    of a Ward dendrogram built on latent space. 
    """
    Z = linkage(latent, method="ward", metric="euclidean")
    # From https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html
    dend = dendrogram(Z, no_plot=True)   # we only need the order
    return dend["leaves"]                # list of ints


trained_model = torch.load("dim_2_3.pt", map_location="cpu")

w = trained_model["w"]
v = trained_model["v"]

# Drug order
drug_order = leaf_order_from_linkage(w.numpy())
# Side‑effect order
se_order   = leaf_order_from_linkage(v.numpy())

# From https://numpy.org/doc/stable/reference/generated/numpy.ix_.html
Y_ordered = Y_np[np.ix_(drug_order, se_order)] 

plt.figure(figsize=(10, 8))
plt.imshow(Y_ordered, aspect="auto", cmap="gray_r", interpolation="nearest")
plt.colorbar(label="Edge (1) / No edge (0)")

plt.title("Ordered adjacency matrix (Y) - Ward hierarchy")
plt.xlabel("Side-effects (ordered by hierarchical clustering)")
plt.ylabel("Drugs (ordered by hierarchical clustering)")

# No tick marks, please.
plt.xticks([]) 
plt.yticks([])

plt.tight_layout()
plt.savefig("heatmap.png", dpi=300, bbox_inches="tight")
print(f"Ordered adjacency matrix saved.")