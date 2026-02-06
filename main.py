import pandas as pd
import torch

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


def main():
    print("Total number of drugs = ", total_no_of_drugs)
    print("Total number of side-effects = ", total_no_of_SE)
    print("Shape of Y matrix = ", Y.shape)
    
if __name__ == "__main__":
    main()