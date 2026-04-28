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

# Import modules

import os

import pubchempy as pcp
import numpy as np
import pandas as pd

from pathlib import Path

from rdkit.Chem import MolFromInchi
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import ExactMolWt

from src.RepoRT_data_processing.RepoRT_get_raw_data import merge_complete_file

# Parameters

RAW_REPORT_FILE = os.path.join (".", "data", "RepoRT", "raw_data", "raw_rt_data.tsv")
PATH2FILE = os.path.join (".", "data", "moldescs.tsv")
PATH2COMPLETE_FILE = os.path.join (".", "data", "complete_moldesc.tsv")

# Define the functions
def _pubchem_moldesc_query (path2raw_report=RAW_REPORT_FILE,
                            save_file = PATH2FILE) -> None:
    """
        This function can be used as the first part to complete the initial moldesc fetching.
        Finally, this function writes a .tsv files containing 3 columns: Inchi, monoisotopic_mass, xlogp.
        This function has been defined as PubChem uses a slightly better algorithm (XLogP 3.0) than RDkit.
    """
    # Check if the raw RepoRT datafile exists first, if not it is built.
    if not Path(path2raw_report).exists():
        merge_complete_file()
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


def complete_moldesc_query (initial_file = PATH2FILE,
                            path2res = PATH2COMPLETE_FILE)-> None:
    """
        This function requires an initial file, containing inchi and moldescs fetched with PubChem.
        If not exists, it will be built.
        This completes the moldesc search by using RDkit, and writes another .tsv file.
    """

    # Check for the initial dataframe
    if not Path (initial_file).exists():
        _pubchem_moldesc_query()

    df = pd.read_csv (initial_file, sep="\t")

    for index, row in df.iterrows():
        if pd.isnull (row["mono_iso_mass"]) or pd.isnull (row["xlogp"]):
            inchi = row["inchi"]
            mol = MolFromInchi(inchi, sanitize=False)
            mol_wt = ExactMolWt (mol)
            logp = MolLogP(mol)
            df.loc[index, "mono_iso_mass"] = round(mol_wt, 9)
            df.loc[index, "xlogp"] = round(logp, 1)
        else:
            continue
    df.to_csv (path2res, sep = "\t", index=False)

if __name__ == "__main__":
    complete_moldesc_query()

