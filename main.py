import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt 

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

latent_space_dimension = 3
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


def distance_in_latent_space(w, v):
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

    

def negative_loglikelihood(Y, psi, omega, w, v):
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



def main():
    print("Total number of drugs = ", total_no_of_drugs)
    print("Total number of side-effects = ", total_no_of_SE)
    print("Shape of Y matrix = ", Y.shape)
    print("\n--- Parameter initialisation summary ---")
    print(f"psi shape          : {psi.shape}")
    print(f"omega shape        : {omega.shape}")
    print(f"w (drug latents) shape : {w.shape}")
    print(f"v (SE latents) shape   : {v.shape}")
    print("----------------------------------------\n")
    print("distance tensor shape = ", distance_in_latent_space(w, v).shape)
    print("-loglikelihood = ", negative_loglikelihood(Y, psi, omega, w, v))
    
    
    ### Convert parameters to learnable tensors. ###
    #  Then the optimiser knows to compute gradients.
    
    psi_l = torch.nn.Parameter(psi.clone())
    omega_l = torch.nn.Parameter(omega.clone())
    w_l = torch.nn.Parameter(w.clone())
    v_l = torch.nn.Parameter(v.clone())

    ### End of parameter convertion. ###

    optimizer = torch.optim.Adam([psi_l, omega_l, w_l, v_l], lr=0.01, weight_decay=1e-4)
    
    
    ### Training loop. ###
    
    number_of_iterations = 200
    loss_history = []  # Store the NLL at each step
    
    for it in range(1, number_of_iterations + 1):
        optimizer.zero_grad()   # Clear old grads.
        loss = negative_loglikelihood(Y, psi_l, omega_l, w_l, v_l)
        loss.backward()  # Back‑propagate.
        optimizer.step() # Take a gradient step.

        loss_history.append(loss.item())          

        if it % 20 == 0 or it == 1:               
            print(f"Iter {it:03d} - NLL = {loss.item():.4f}")
            
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
            
            
if __name__ == "__main__":
    main()