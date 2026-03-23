"""
Name: perform_cc_scaffold_split.py
Author: Yixi Zhang
Date: March 2026
Version: 1.0.
Description: Contains functions for performing splitting by both chromatography condition and Murcko Scaffold, this is, there should not be molecules of, for example,
directory 0001 and murcko-scaffold ccccccc1N in different datasets. This is to avoid data leakage and test if a MPNN can really be projectable.
"""
import random

#IMPORT MODULES

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.training.RepoRT.with_SMRT_ds.scaffold_split.perform_scaffold_split import *

# DEFINE THE INPUT PATHS

ms_train = Path("./data/processed_RepoRT/with_SMRT_ds/ms_split_data/train_data.tsv")
ms_val = Path("./data/processed_RepoRT/with_SMRT_ds/ms_split_data/val_data.tsv")
ms_test = Path("./data/processed_RepoRT/with_SMRT_ds/ms_split_data/test_data.tsv")
output_dir = "./data/processed_RepoRT/with_SMRT_ds/cc_ms_split_data/"
def cc_ms_split (ms_train_file = ms_train, ms_val_file = ms_val, ms_test_file = ms_test, save_dir =output_dir):
    # Check the input files.
    if not (ms_train_file.exists() and ms_val_file.exists() and ms_test_file.exists()):
        print ("Creating the Scaffold split files...")
        ms_split()

    print ("Reading the input files...")
    ms_train_df = pd.read_csv(ms_train_file, sep='\t')
    ms_val_df = pd.read_csv(ms_val_file, sep='\t')
    ms_test_df = pd.read_csv(ms_test_file, sep='\t')
    total_input_df = pd.concat ([ms_train_df, ms_val_df, ms_test_df], ignore_index=True) #This df is important for splitting
    index_array = np.unique (total_input_df ["dir_id"]) #This ensures that it contains all repo indices.

    print ("Checking for the output dir...")
    os.makedirs(save_dir, exist_ok=True)

    print ("Splitting by ms_smiles the input data...")
    #First splitting by ms_smiles
    ms_smiles_array = np.unique (total_input_df ["ms_smiles"])
    np.random.seed (42)
    np.random.shuffle (ms_smiles_array)
    train_ms_smiles, val_ms_smiles, test_ms_smiles = [],[],[]
    train_size = val_size = test_size = 0
    threshold = round (total_input_df.shape [0]/3) #Make sure that there are approx. 1/3 molecules in the first split.
    for ms_smiles in ms_smiles_array:
        temp_df = total_input_df[total_input_df["ms_smiles"] == ms_smiles]
        if train_size + temp_df.shape [0]< threshold:
            train_ms_smiles.append(ms_smiles)
            train_size += temp_df.shape[0]
        elif val_size + temp_df.shape [0]< threshold:
            val_ms_smiles.append (ms_smiles)
            val_size += temp_df.shape[0]
        else:
            test_ms_smiles.append(ms_smiles)
            test_size += temp_df.shape[0]
    train_ms_split_df = total_input_df[total_input_df["ms_smiles"].isin(train_ms_smiles)]
    val_ms_split_df = total_input_df[total_input_df["ms_smiles"].isin (val_ms_smiles)]
    test_ms_split_df = total_input_df[total_input_df["ms_smiles"].isin (test_ms_smiles)]

    print("Performing second splitting by dir_ids...")
    # Perform dir_id splitting
    np.random.shuffle (index_array)
    train_indices, test_indices = train_test_split (index_array, test_size=0.1, random_state=42)
    train_indices, val_indices = train_test_split (train_indices, test_size=0.1111, random_state=42)
    #This just works don't change anything pls. The size of the resulting train set is 79% and both val and test set is around 10%
    final_train_df = train_ms_split_df[train_ms_split_df["dir_id"].isin (train_indices)]
    final_val_df = val_ms_split_df[val_ms_split_df["dir_id"].isin (val_indices)]
    final_test_df = test_ms_split_df[test_ms_split_df["dir_id"].isin (test_indices)]

    print (f"Saving result files to {save_dir}")
    final_train_df.to_csv (f"{save_dir}train_data.tsv", sep='\t', index=False)
    final_val_df.to_csv (f"{save_dir}val_data.tsv", sep='\t', index=False)
    final_test_df.to_csv(f"{save_dir}test_data.tsv", sep='\t', index=False)
    return




