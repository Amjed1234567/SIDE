import pandas as pd
import torch
import numpy as np

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
    
if __name__ == "__main__":
    main()