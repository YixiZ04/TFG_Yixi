"""
Name: SMRT_get_moldescs.py
Author: Yixi Zhang
Date: March 2026
Version: 1.1.
Usage: Using Pubchempy get two molecular descriptors:
    1. Monoisotopic molecular weight
    2. XlogP. Metric for polarity of the molecule.
Pubchempy API is preferred because its xlogp values are more accurate to the experimental ones. However the main drawback is that Pubchempy is much slower than RDkit.
This Script can easily last close to a day to finish getting the results.
Update: Now this script contains functions to get those molecular descriptors mentioned above using Inchi instead of CID. Also it builds a dataframe containing 3 columns:
    1. "inchi.std"
    2. "mono_iso_mass"
    3. "xlogp"
it will be saved as a tsv file in ./data/with_extra_mol_desc/.
Molecular descriptors for both RepoRT and SMRT will be fetched from this df using the inchi as the query input; also it avoids fetching >1 times molecular descriptors for a molecule
whose information has already been fetched, thus accelerating the process.
Note: this dataframe has already been built and come with this repository. Only need to run this script if there were any updates.
"""
# 0. Import modules
import pubchempy as pcp
import numpy as np
import pandas as pd
from pathlib import Path
from src.get_raw_data.RepoRT_get_all_data import get_raw_datatable


# 1. Define the functions
def build_mol_desc_dict_from_inchi (path2raw_report="./data/no_extra_mol_desc/RepoRT_complete_data.tsv", save_file ="./data/with_extra_mol_desc/extra_mol_descs.tsv"):
    """
    Inputs: path2raw_report = The path to the RepoRT raw file (containing all information), the path to the file for saving the dataframe.
    Output: The dataframe saved in the indicated path.
    """
    # Check if the raw RepoRT datafile exists first, if not it is built.
    if not Path (path2raw_report):
        get_raw_datatable()
    df = pd.read_csv(path2raw_report, sep="\t")
    inchi_array = df.loc[:, "inchi.std"].values
    mono_mw_dict = {}
    xlogp_dict = {}
    for inchi in inchi_array:
        if inchi not in mono_mw_dict.keys():
            try:
                print(f"Fetching mol descs for molecule {inchi}")
                molecule = pcp.get_compounds(str(inchi), namespace="inchi")
                mono_mw = molecule[0].monoisotopic_mass
                xlogp = molecule[0].xlogp
                mono_mw_dict[inchi] = mono_mw
                xlogp_dict[inchi] = xlogp
            except:  # This except clause is for not found molecules in PubChem.
                mono_mw_dict[inchi] = np.nan
                xlogp_dict[inchi] = np.nan
        else:  # If the molecule has already been fetched.
            print(f"The molecule: {inchi}'s has already been fetched. It will be skipped.")
            continue
    #Build the final dataframe usign the information fetched. Both dictionary have the same dimension, as np.nan has been used for filling any missing values.
    final_df = pd.DataFrame({"inchi": mono_mw_dict.keys(),
                             "mono_iso_mass": mono_mw_dict.values(),
                             "xlogp": xlogp_dict.values()})
    final_df.to_csv (save_file, sep = "\t", index=False)
    return

def add_columns2df (path2dataset, dataset, save_path = "./data/with_extra_mol_desc/",path2moldesc = "./data/with_extra_mol_desc/extra_mol_descs.tsv"):
    """
    Inputs: Path to the dataset (SMRT, RepoRT_all, RepoRT_only_mol). Which dataset should be used (SMRT,RepoRT_all, RepoRT_only_mol) and the path used for saving the df of molecular descriptors.
    Output: An updated dataframe containing the molecular descriptors.
    """
    # By running this function if the extra_mol_desc.tsv file is not found it will be built, but should not be the case.
    if not Path (path2moldesc).exists():
        build_mol_desc_dict_from_inchi()
    moldesc_df = pd.read_csv(path2moldesc, sep="\t")
    save_filename = f"{save_path}{dataset}_extra_mol_descs.tsv"
    if dataset == "SMRT":
        data_df = pd.read_csv(path2dataset, sep=";")
        final_df = pd.merge (data_df, moldesc_df, on="inchi", how = "inner")
        final_df = final_df.dropna ()
        final_df.to_csv (save_filename, sep = "\t", index = False)
        return
    elif dataset == "RepoRT_all" or dataset == "RepoRT_only_mol":
        data_df = pd.read_csv(path2dataset, sep="\t")
        temp_df = pd.merge(data_df, moldesc_df, left_on = "inchi.std", right_on="inchi", how = "inner")
        # "Cut and paste action"
        temp_mass_serie = temp_df.pop ("mono_iso_mass")
        temp_xlog_serie = temp_df.pop ("xlogp")
        position = temp_df.columns.get_loc ("rt_s")
        temp_df.insert (position + 1, "mono_iso_mass", temp_mass_serie)
        temp_df.insert (position + 2, "xlogp", temp_xlog_serie)
        final_df = temp_df.dropna (subset=["mono_iso_mass", "xlogp"])
        final_df = final_df.drop (columns = ["inchi"])
        final_df.to_csv (save_filename, sep = "\t", index = False)
    else:
        print (f"The dataset introduced: {dataset} is not correct")
        return

# This could be used if wanted to get the complete RepoRT raw dataset with mol descs.
# if __name__ == "__main__":
#     add_columns2df("./data/no_extra_mol_desc/RepoRT_complete_data.tsv", "RepoRT_all")