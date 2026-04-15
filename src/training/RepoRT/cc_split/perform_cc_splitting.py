"""
Name: perform_cc_splitting.py
Author: Yixi Zhang
Date: March 2026
Version: 1.2.
Description: Contains all the functions for performing CC (chromatography conditions) splitting.
Update (1.1.): Implemented the new version of data_processing.py with the filtered datafile.
And random seed shuffle has also been implemented as the datasets' size does not differ so much.
Update (1.2.): Adapted to the version 1.2. of data_processing.py, as now this splitting function is more flexible.
"""

# IMPORT MODULES
import pandas as pd
import numpy as np
import os
from pathlib import Path
from src.RepoRT_data_processing.RepoRT_processing import get_processed_df_from_raw

# DEFINE THE FUNCTION
def cc_split (input_path, output_dir, drop_smrt, apply_low_grad_filter):
    """
        Splitting the processed datafile (input_path) given by chromatography conditions.
        The train, test and val tsv files will be saved in the output_dir
        If the processed RepoRT input file does not exist, it will be created using drop_smrt (Boolean) and apply_low_grad_filter (Boolean).
    """
    print ("Checking the input file...")
    file = Path(input_path)
    if file.exists():
        print ("THe input file exists!")
    else:
        print ("The input file does not exist, creating it...")
        get_processed_df_from_raw (drop_smrt=drop_smrt,
                                   down_grad_filter=apply_low_grad_filter,
                                   )

    print ("Getting the input DataFrame...")
    df = pd.read_csv (input_path, sep = "\t",)

    print ("Making the saving directory...")
    os.makedirs (output_dir, exist_ok = True)

    #Main process
    dir_ids = np.unique (df["dir_id"])
    np.random.seed (42)
    np.random.shuffle (dir_ids)
    print ("Splitting into differente sets")
    train_ids, val_ids, test_ids = [], [], []
    train_size = test_size = val_size = 0
    for dir_id in dir_ids:
        temp_df = df[df["dir_id"] == dir_id]
        if train_size + temp_df.shape [0]< 0.8*df.shape[0]:
            train_ids.append(dir_id)
            train_size += temp_df.shape[0]
        elif val_size + temp_df.shape [0]< 0.1*df.shape[0]:
            val_ids.append(dir_id)
            val_size += temp_df.shape[0]
        else:
            test_ids.append(dir_id)
            test_size += temp_df.shape[0]
    train_dset = df[df["dir_id"].isin (train_ids)]
    val_dset = df[df["dir_id"].isin (val_ids)]
    test_dset = df[df["dir_id"].isin (test_ids)]
    train_file = output_dir + "train_data.tsv"
    val_file = output_dir + "val_data.tsv"
    test_file = output_dir + "test_data.tsv"
    print (f"Exporting files to {output_dir}")
    train_dset.to_csv (train_file, sep = "\t", index = False)
    val_dset.to_csv (val_file, sep = "\t", index = False)
    test_dset.to_csv (test_file, sep = "\t", index = False)

