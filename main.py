import pandas as pd
import numpy as np

data_file = "Test.xlsx" 
drug_column_index = 0 
SE_column_index = 1 # Side effects.
   
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
Y = np.zeros((total_no_of_drugs, total_no_of_SE))
for row in data_list_2D:
    i = drug_number_dict[row[0]]
    j = SE_number_dict[row[1]]
    Y[i][j] = 1        

def main():
    print("This is the Y matrix: ")
    print(Y)
    print("Dicts: ")
    print(drug_number_dict)
    print(SE_number_dict)
    
if __name__ == "__main__":
    main()