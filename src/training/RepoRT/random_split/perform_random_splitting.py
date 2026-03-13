"""
Name: perform_random_splitting.py
Author: Yixi Zhang
Date: March 2026
Version: 1.0
Usage: Separate the df in three sets: training, validation and testing totally random:
        4.1. The train set contains 80% of molecules from each repository||chromatography condition.
        4.2. The val set contains 10% of molecules form each repository||chromatography condition.
        4.3. The test set contains 10% of molecules from each repository||chromatography condition.
A random seed (42) is set, so the random splitting should be reproducible unless this is changed.
The function defined here will create the random split input datafile in: ./data/processed_data/random_split_data/
If the processed datafile does not exist, it will be created from the raw datafile. In the same way, if the raw datafile does not exist, it will be created.
(See ./src/process_RepoRT_data/data_processing.py)
"""

# IMPORT MODULES
import pandas as pd
from sklearn.model_selection import train_test_split
from src.process_RepoRT_data.data_processing import get_processed_df_from_raw
import os
from pathlib import Path

def split_train_val_test (input_path = "./data/processed_RepoRT/complete_treated_data.tsv",path2res ="./data/processed_RepoRT/random_split_data/"):
    """
    Uses the scaled_complete_df to perform train_test_split.
    Random_state set to 42 for consistency.
    """
    file = Path (input_path)
    if file.exists():
        print ("The input file is correct!")
    else:
        print ("Getting the processed datafile...")
        get_processed_df_from_raw()
    df = pd.read_csv(input_path, sep = "\t")

    print ("Checking the output dir...")
    os.makedirs(path2res, exist_ok = True)
    train_list = []
    test_list = []
    val_list = []
    for index in range (1, 440):
        temp_df = df[df["dir_id"] == index]
        if temp_df.shape[0] != 0:
            train_df, test_df = train_test_split(temp_df, test_size = 0.1, random_state = 42)
            test_list.append(test_df)
            train_df, val_df = train_test_split(train_df, test_size = 0.1111, random_state = 42)
            train_list.append(train_df)
            val_list.append(val_df)
        else:
            continue
    train_set = pd.concat (train_list, ignore_index=True)
    val_set = pd.concat (val_list, ignore_index=True)
    test_set = pd.concat (test_list, ignore_index=True)
    train_filename = path2res + "train_data.tsv"
    test_filename = path2res + "test_data.tsv"
    val_filename = path2res + "val_data.tsv"
    train_set.to_csv (train_filename, sep = "\t", index = False)
    test_set.to_csv (test_filename, sep = "\t", index = False)
    val_set.to_csv (val_filename, sep = "\t", index = False)
    print (f"Successfully split the processed datafile randomly and the datafiles will be saved in {path2res}")


