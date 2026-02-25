"""
"""

# 0. Import modules

import pubchempy as pcp
import pandas as pd
import numpy as np

# 1. Define a function to get mol_desc

def get_mol_desc_from_inchi (inchi_array):
    """ 
    Input: an inchi array.
    Output: molecular weight and xlogp arrays
    """
    mol_weight_array = []
    xlop_array = []
    for inchi in inchi_array:
        try:
            print (f"Fetching mol desc for the molecule: {inchi}:")
            molecule = pcp.get_compounds(str(inchi), namespace="inchi")
            mol_weight = np.float16(molecule[0].molecular_weight)
            xlogp = np.float16 (molecule[0].xlogp)
            mol_weight_array.append(mol_weight)
            xlop_array.append(xlogp)
        except:
            print(f"Cannot fetch mol desc for the molecule: {inchi}.")
            mol_weight_array.append(np.nan)
            xlop_array.append(np.nan)

    return mol_weight_array, xlop_array

# 2. Load the data

path2file = "/home/user/YixiTFG/TFG_Yixi/RepoRT_trials/RepoRT_data/RepoRT_data_clean.tsv"
df = pd.read_csv(path2file, sep="\t")
inchi_array = df.loc [:,"inchi"].values

mol_weight_array, xlop_array = get_mol_desc_from_inchi (inchi_array)

df ["mol_weight"] = mol_weight_array
df ["xlogp"] = xlop_array
filename = "/home/user/YixiTFG/TFG_Yixi/RepoRT_trials/RepoRT_mol_desc_data/RepoRT_mol_desc.tsv"
df.to_csv (filename, sep="\t", index=False)
