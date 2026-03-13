"""
Name: SMRT_get_moldescs.py
Author: Yixi Zhang
Date: March 2026
Usage: Using Pubchempy get two molecular descriptors:
    1. Monoisotopic molecular weight
    2. XlogP. Metric for polarity of the molecule.
Pubchempy API is preferred because its xlogp values are more accurate to the experimental ones. However the main drawback is that Pubchempy is much slower than RDkit.
This Script can easily last close to a day to finish getting the results.
"""
# 0. Import modules
import pubchempy as pcp
import numpy as np
import pandas as pd
import sys
import os

# 1. Define the functions
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
            molecular_weight = np.float64 (compound.monoisotopic_mass)
            xlogp = np.float64 (compound.xlogp)
            mol_weight_array.append(molecular_weight)
            xlogp_array.append (xlogp)
        except:
            print (f"The molecule with cid {str(cid)} can not be found. NA values will be added.")
            mol_weight_array.append (np.nan)
            xlogp_array.append (np.nan)
    return np.array(mol_weight_array), np.array(xlogp_array)


if __name__ == "__main__":
    # Get the CID array.
    df = pd.read_csv ("./data/no_extra_mol_desc/SMRT_data.csv", sep = ";")
    cids_array = df ["pubchem"]

    #Get the molecular descriptors
    mol_weights, xlogps = get_mol_desc_from_cid (cids_array)

    #Create a new Dataset with this information

    df ["monoisotopic_mass"] = mol_weights
    df ["xlogp"] = xlogps

    # Clean na values. With SMRT can directly applay dropna method as originally it has no "nan" values.
    datafile = df.dropna()

    # Export to a tsv file
    path2dir = "./data/with_extra_mol_desc/"
    os.makedirs(path2dir, exist_ok=True)
    filename = path2dir+ "SMRT_extra_moldesc_data.tsv"
    datafile.to_csv (filename, sep="\t", index=False)
    sys.exit (0)