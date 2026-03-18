"""
Name: perform_cc_splitting.py
Author: Yixi Zhang
Date: March 2026
Version: 1.1.
Description: Contains all the functions for performing CC (chromatography conditions) splitting.
Update: Implemented the new version of data_processing.py with the filtered datafile.
And random seed shuffle has also been implemented as the datasets' size does not differ so much.
"""

# IMPORT MODULES
import pandas as pd
import numpy as np
import os
from pathlib import Path
from src.process_RepoRT_data.data_processing import get_processed_df_from_raw


#DEFINE VARIABLES
# input_path = "./data/processed_RepoRT/complete_treated_data.tsv" #Uncomment this line if want to use the complete data.
input_path = "./data/processed_RepoRT/filtered_treated_data.tsv"
output_dir = "./data/processed_RepoRT/cc_split_data/"
complete = False # Set to True if want to use the data without filtering

# DEFINE THE FUNCTION
def cc_split (input_path = input_path, output_dir = output_dir, complete=complete):
    print ("Checking the input file...")
    file = Path(input_path)
    if file.exists():
        print ("THe input file exists!")
    else:
        print ("The input file does not exist, creating it...")
        get_processed_df_from_raw (complete=complete)

    print ("Getting the input DataFrame...")
    df = pd.read_csv (input_path, sep = "\t",)

    print ("Making the saving directory...")
    os.makedirs (output_dir, exist_ok = True)

    #Main process
    dir_ids = np.unique (df["dir_id"])
    if not complete:
        np.random.seed (42)
        np.random.shuffle (dir_ids)
    print ("Splitting into differente sets")
    train_ids = []
    val_ids = []
    test_ids = []
    train_size = 0
    test_size = 0
    val_size = 0
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

if __name__ == "__main__":
    cc_split ()
    a = pd.read_csv (output_dir+"train_data.tsv", sep = "\t")
    b = pd.read_csv (output_dir+"val_data.tsv", sep = "\t")
    c = pd.read_csv (output_dir+"test_data.tsv", sep = "\t")
    d = pd.read_csv (input_path, sep = "\t")