"""
Name: perform_scaffold_split.py
Author: Yixi Zhang
Date: March 2026
Version: 1.2.
Description: Based on RDkit and Bemis-Murcko Scaffold to avoid data leakage, this is, to make sure each dataset (train/test/val) has different scaffolds.
Update (1.1.): Implemented the new version of data_processing.py with the filtered datafile.
Update (1.2.): Adapted to the version 1.2. of data_processing.py, as now this splitting function is more flexible.
"""

#IMPORT MODULES
from pathlib import Path
import os
import pandas as pd
import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold
from src.process_RepoRT_data.data_processing import get_processed_df_from_raw

def ms_split (input_path, output_dir, drop_smrt, apply_upthreshold):
    """
    Input: Path to processed DataFrame and Directory to save result files.
    Outputs: Saves the train/val/test dsets in .tsv format inside the saving directory.
    """
    if not Path(input_path).exists():
        print ("The input file does not exist. Creating it...")
        get_processed_df_from_raw (drop_smrt=drop_smrt,
                                   apply_upthreshold=apply_upthreshold,
                                   complete=False)

    print ("Reading the input file...")
    df = pd.read_csv (input_path, sep='\t')

    print ("Making the output directory...")
    os.makedirs(output_dir, exist_ok = True)

    print ("Getting Murcko SMILES from SMILES...")
    smiles = df.loc [:, "smiles.std"].values
    ms_smiles = np.array ([ MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smile) for smile in smiles ])
    position = df.columns.get_loc ("smiles.std")
    df.insert (position + 1,"ms_smiles", ms_smiles)
    final_df = df[df["ms_smiles"] != ""]

    print (f"Getting train/val/test dsets and saving them in {output_dir}...")
    ms_smiles_classes = np.unique (final_df["ms_smiles"])
    np.random.seed (42)
    np.random.shuffle (ms_smiles_classes)
    train_dset, val_dset, test_dset = [], [], []
    train_size, val_size, test_size = 0, 0, 0

    # The main loop to ensure that the sizes for each dset is close to (0.8,0.1,0.1)
    for ms_smiles in ms_smiles_classes:
        temp_df = df[df["ms_smiles"] == ms_smiles]
        if train_size + temp_df.shape [0] < 0.8*final_df.shape[0]:
            train_dset.append (temp_df)
            train_size+=temp_df.shape [0]
        elif val_size + temp_df.shape [0] < 0.1*final_df.shape[0]:
            val_dset.append (temp_df)
            val_size+=temp_df.shape [0]
        else:
            test_dset.append (temp_df)
            test_size+=temp_df.shape [0]

    # Get the dataframes from the list of dataframes
    train_dset = pd.concat (train_dset, ignore_index = True)
    val_dset = pd.concat (val_dset, ignore_index = True)
    test_dset = pd.concat (test_dset, ignore_index = True)
    complete_dset = pd.concat ([train_dset, val_dset, test_dset], ignore_index = True)

    #Export those dataframes to a .tsv file in the output_dir
    train_file = output_dir + "train_data.tsv"
    val_file = output_dir + "val_data.tsv"
    test_file = output_dir + "test_data.tsv"
    complete_file = output_dir + "ms_complete_data.tsv"
    train_dset.to_csv (train_file, sep = "\t", index = False)
    val_dset.to_csv (val_file, sep = "\t", index = False)
    test_dset.to_csv (test_file, sep = "\t", index = False)
    complete_dset.to_csv (complete_file, sep = "\t", index = False)

