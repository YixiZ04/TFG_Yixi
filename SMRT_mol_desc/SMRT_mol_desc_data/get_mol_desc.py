"""
Author: Yixi Zhang
This .py file focuses on get mol descriptors from PubChem:
    *Molecular weight (g/mol)
    *XLogP (adimensional) ==> Informs the polarity of the molecule
This bases on the PubChem API (PubChemPy)
"""

import pubchempy as pcp
import numpy as np
import pandas as pd

def get_mol_desc_from_cid (cid_array):
    """
    Input: an array containing pubchem ID (CID). (SMRT Dataset contians this information)
    Output: 2 arrays containing molecular weight and XlogP values.
    If a molecule info is not present, "NA" values is added to the array
    """

    mol_weight_array = []
    xlogp_array = []

    # Main loop for the
    for cid in cid_array:
        try:
            print (f"Getting molecule with cid {str(cid)} ...")
            compound = pcp.Compound.from_cid (cid)
            molecular_weight = np.float16 (compound.molecular_weight)
            xlogp = np.float16 (compound.xlogp)
            mol_weight_array.append(molecular_weight)
            xlogp_array.append (xlogp)
        except:
            print (f"The molecule with cid {str(cid)} can not be found. NA values will be added.")
            mol_weight_array.append (np.nan)
            xlogp_array.append (np.nan)
    return np.array(mol_weight_array), np.array(xlogp_array)

#Get cid value

path_to_file = "/home/user/YixiTFG/TFG_Yixi/SMRT_trials/SMRT_Data/SMRT_dataset.csv"
datafile = pd.read_csv(path_to_file, sep=";")
cids_array = datafile ["pubchem"]

#Get mol descs

mol_weights, xlogps = get_mol_desc_from_cid (cids_array)

#Create a new Dataset with this information

datafile ["mol_weight"] = mol_weights
datafile ["xlogp"] = xlogps

# Clean na values

datafile = datafile.dropna()

# Export to a new csv file

path_2_new_datafile = "/home/user/YixiTFG/TFG_Yixi/SMRT_mol_desc/SMRT_mol_desc_data/SMRT_mol_desc.csv"
datafile.to_csv (path_2_new_datafile, sep=";",index=False)