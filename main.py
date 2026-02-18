import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import os 
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

### Environment configuration (provided by bsub). ###

#  lsd  – latent‑space dimension (integer).
#  name – base name for the saved model file (string).
DEFAULT_LATENT_DIM = 3                     # Default.
DEFAULT_MODEL_NAME = "trained_model.pt"    # Default.

try:
    latent_space_dimension = int(os.getenv("lsd", DEFAULT_LATENT_DIM))
except ValueError:
    # If the user passes a non‑numeric string we keep the default
    latent_space_dimension = DEFAULT_LATENT_DIM

model_name = os.getenv("name", DEFAULT_MODEL_NAME)

### End of environment configuration (provided by bsub). ###


### Data preparation section ###

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

### End of data preparation section ###


### Initialization of parameters. ###

seed_for_random = 42 # For reproducibility.
torch.manual_seed(seed_for_random) 
np.random.seed(seed_for_random)

# Random effect psi. One value for each drug.
psi = torch.from_numpy(np.random.randn(total_no_of_drugs)).float()
# Random effect psi. One value for each side‑effect.
omega = torch.from_numpy(np.random.randn(total_no_of_SE)).float()

# Latent position w_i for each drug, shape (total_no_of_drugs, latent_space_dimension).
w = torch.from_numpy(np.random.randn(total_no_of_drugs, latent_space_dimension)).float()

# Latent position v_j for each side‑effect, shape (total_no_of_SE, latent_space_dimension).
v = torch.from_numpy(np.random.randn(total_no_of_SE, latent_space_dimension)).float()


### End of initialization of parameters. ###


def distance_in_latent_space(w, v)->float:
    """Calculating distance betwen two points in latent space.

    Args:
        w (torch.tensor): Shape (total_no_of_drugs, latent_space_dimension).
        v (torch.tensor): Shape (total_no_of_SE, latent_space_dimension).
    
    Returns a Tensor of shape (total_no_of_drugs, total_no_of_SE) where entry (i,j) = ||w_i - v_j||
    """
    # Broadcasting: (total_no_of_drugs, 1, latent_space_dimension) - 
    # (1, total_no_of_SE, latent_space_dimension) -> 
    # (total_no_of_drugs, total_no_of_SE, latent_space_dimension)
    difference = w[:, None, :] - v[None, :, :]
    # ord=2 is the Euclidean norm. 
    return torch.linalg.norm(difference, ord=2, dim=2)

    
def negative_loglikelihood(Y, psi, omega, w, v)->torch.tensor:
    """Calculating the negative loglikelihood (NLL).
    
    Args:
        Y (torch.tensor): Shape (total_no_of_drugs, total_no_of_SE) with entries 0/1.
        w (torch.tensor): Shape (total_no_of_drugs, latent_space_dimension).
        v (torch.tensor): Shape (total_no_of_SE, latent_space_dimension).
        psi (torch.tensor): Shape (total_no_of_drugs,)
        omega (torch.tensor): Shape (total_no_of_SE,)
    
    Returns a scalar Tensor (the NLL)
    """
    # Broadcast psi and omega to matrix shape.
    psi_matrix   = psi[:, None]  # (total_no_of_drugs, 1)
    omega_matrix = omega[None, :] # (1, total_no_of_SE)
    
    # Shape of eta is (total_no_of_drugs, total_no_of_SE)
    eta = psi_matrix + omega_matrix - distance_in_latent_space(w, v)
    
    # https://docs.pytorch.org/docs/stable/generated/torch.log1p.html
    one_pair_of_ij = -torch.log1p(torch.exp(eta)) + Y * eta
    
    # Sum over all i, j = NLL.
    return -1*one_pair_of_ij.sum() 


# Calculate the loss per matrix entry. 
def print_progress(it, loss_tensor):
    
    total_entries = total_no_of_drugs * total_no_of_SE
    avg_per_entry = loss_tensor.item() / total_entries
    print(
        f"Iter {it:03d} - NLL = {loss_tensor.item():,.1f} "
        f"(avg per entry = {avg_per_entry:.6f})"
    )
    

def frequency_of_elements(Y_input:torch.tensor)->float:
    """Calculating the frequency of ones and zeros.

    Args:
        Y_input (torch.tensor): Shape (total_no_of_drugs, total_no_of_SE) with entries 0/1.

    Returns:
        float: Two floats. Freqency of ones and zeros.
    """
    total_no_of_elements = total_no_of_drugs * total_no_of_SE
    ones = int(torch.count_nonzero(Y_input)) 
    zeros = total_no_of_elements - ones 
    freq_ones = ones / total_no_of_elements
    freq_zeros = zeros / total_no_of_elements
    
    return freq_ones, freq_zeros


def replace_ones_with_zeros(Y_original:torch.tensor, pct:float)->torch.tensor:
    """Randomly a certain fraction (pct) of the ones will become zeros.

    Args:
        Y_original (torch.tensor): Shape (total_no_of_drugs, total_no_of_SE) with entries 0/1.
        pct (float): The fraction of ones to become zeros. 

    Returns:
        torch.tensor: Shape (total_no_of_drugs, total_no_of_SE) with entries 0/1.
    """   
    out = Y_original.clone()
    # Find [row, column] for each of the ones in the matrix. 
    index = torch.nonzero(out == 1, as_tuple=False)   # shape (n_ones, 2)
    n_ones = index.shape[0]  # Total number of ones.
    # pct is the fraction we want to change to zeros.
    n_drop = int(pct * n_ones)  # Absolute no.of ones to turn into zeros.
    
    # Randomly choose ones to be replaced.
    rng = torch.Generator(device=out.device) 
    perm = torch.randperm(n_ones, generator=rng, device=out.device)
    
    chosen_positions = index[perm[:n_drop]]  # shape (n_drop, 2)
    out[chosen_positions[:, 0], chosen_positions[:, 1]] = 0
    
    return out


def edge_probability_matrix(psi, omega, w, v):
    """Returns a matrix where each entry [i,j] is 
        the model's estimate that an edge exists.

    Args:
        w (torch.tensor): Shape (total_no_of_drugs, latent_space_dimension).
        v (torch.tensor): Shape (total_no_of_SE, latent_space_dimension).
        psi (torch.tensor): Shape (total_no_of_drugs,)
        omega (torch.tensor): Shape (total_no_of_SE,)

    Returns:
        torch_tensor: Probability matrix.
    """
    eta = psi[:, None] + omega[None, :] - distance_in_latent_space(w, v)
    return torch.sigmoid(eta)


def pr_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute the area under the Precision-Recall curve.

    Args:
        y_true (np.ndarray): Ground-truth binary labels (0 or 1), flattened.
        y_score (np.ndarray): Predicted probabilities (output of the model), flattened.

    Returns:
        float: The PR-AUC.
    """
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
    # Arrays of precision, recall, and thresholds.
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
    return auc(recall, precision)


### Count ones and zeros in Y and Y_perturbed. ###

one, zero = frequency_of_elements(Y)
Y_perturbed = replace_ones_with_zeros(Y, 0.10)
one_perturbed, zero_perturbed = frequency_of_elements(Y_perturbed)

### End of counting ones and zeros in Y and Y_perturbed. ###



def main():
    print("\n--- Data information ---")
    print("Total number of drugs = ", total_no_of_drugs)
    print("Total number of side-effects = ", total_no_of_SE)
    print("----------------------------------------\n")
    print("\n")
    
    print("\n--- Y matrix data ---")
    print("Shape of Y matrix = ", Y.shape)
    print("Frequency of ones = ", one)
    print("Frequency of zeros = ", zero)
    print("----------------------------------------\n")
    print("\n") 
    
    print("\n--- Perturbed Y matrix data ---")    
    print("Shape of perturbed Y matrix = ", Y_perturbed.shape)
    print("Frequency of ones = ", one_perturbed)
    print("Frequency of zeros = ", zero_perturbed)
    print("----------------------------------------\n")
    print("\n")             
    
    print("\n--- Parameter initialisation summary ---")
    print(f"psi shape          : {psi.shape}")
    print(f"omega shape        : {omega.shape}")
    print(f"w (drug latents) shape : {w.shape}")
    print(f"v (SE latents) shape   : {v.shape}")
    print("----------------------------------------\n")
    print("\n")
    
    print("\n--- Other information ---")
    print("distance tensor shape = ", distance_in_latent_space(w, v).shape)
    print("Latent space dimension", latent_space_dimension)
    print("----------------------------------------\n")
    print("\n")
    
    
    ### Convert parameters to learnable tensors. ###
    #  Then the optimiser knows to compute gradients.
    
    psi_l = torch.nn.Parameter(psi.clone())
    omega_l = torch.nn.Parameter(omega.clone())
    w_l = torch.nn.Parameter(w.clone())
    v_l = torch.nn.Parameter(v.clone())

    ### End of parameter convertion. ###

    optimizer = torch.optim.Adam([psi_l, omega_l, w_l, v_l], lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)
    
    ### Training loop. ###
    
    number_of_iterations = 200
    loss_history = []  # Store the NLL at each step
    
    for it in range(1, number_of_iterations + 1):
        optimizer.zero_grad()   # Clear old grads.
        loss = negative_loglikelihood(Y, psi_l, omega_l, w_l, v_l)
        loss.backward()  # Back‑propagate.
        optimizer.step() # Take a gradient step.
        scheduler.step()

        loss_history.append(loss.item())          

        if it % 20 == 0 or it == 1:               
            print_progress(it, loss)
            
    ### End of training loop. ###
     
     
    ### Plot loss vs. iteration. ###
     
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, number_of_iterations + 1), loss_history, marker='o', markersize=3)
    plt.title("Negative LogLikelihood vs. Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Negative LogLikelihood")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    output_path = "negative_loglikelihood.png"   
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
     
    ### End of plot. ###
    
    ### Predict probabilities for entries. ###
    
    prob_matrix = edge_probability_matrix(psi_l.detach(), omega_l.detach(), w_l.detach(), v_l.detach())

    # Flatten everything once – this makes indexing easy
    prob_flat = prob_matrix.reshape(-1).cpu().numpy()          # model scores
    Y_flat    = Y.reshape(-1).cpu().numpy()                    # ground‑truth (original)
    Yp_flat   = Y_perturbed.reshape(-1).cpu().numpy()         # perturbed version
    
    ### End of predict probabilities for entries. ###
    
    
    ### Build the evaluation set. ###
    
    # Keep only entries that are zero in the perturbed matrix.
    # Label = 1 for false‑zeros (original Y = 1)
    # Label = 0 for true‑zeros  (original Y = 0)

    mask_zero_in_perturbed = (Yp_flat == 0) # Boolean mask (True where entry is zero).
    labels = np.where(Y_flat == 1, 1, 0) # 1 = original edge, 0 = no edge.
    
    # Take the array labels and keep only those entries whose 
    # corresponding position in mask_zero_in_perturbed is True.
    eval_labels = labels[mask_zero_in_perturbed]   
    eval_scores = prob_flat[mask_zero_in_perturbed]   # model probabilities for those entries.
    
    ### End of build the evaluation set. ###
    
    ### ROC-AUC and PR_AUC section ###
    
    # https://www.scikit-yb.org/en/latest/api/classifier/rocauc.html
    roc_auc = roc_auc_score(eval_labels, eval_scores)
    pr_auc  = auc(*precision_recall_curve(eval_labels, eval_scores)[1:])    
    
    print("\n=== Zero-Prediction Evaluation ===")
    print(f"ROC-AUC = {roc_auc:.4f}")
    print(f"PR-AUC (Average Precision) = {pr_auc:.4f}")
    
    ### End of ROC AUC and PR-AUC ###
    
    
    ###  Save the trained model. ###
    
    checkpoint = {
        "latent_dim": latent_space_dimension,
        "psi":    psi_l.detach().cpu(),
        "omega":  omega_l.detach().cpu(),
        "w":      w_l.detach().cpu(),
        "v":      v_l.detach().cpu(),
    }
    torch.save(checkpoint, model_name)
    print(f"Model checkpoint saved as '{model_name}'")
    
    ###  End of save the trained model. ###
            
            
if __name__ == "__main__":
    main()