"""
Name: perform_random_splitting.py
Author: Yixi Zhang
Date: March 2026
Version: 1.2.
Usage: Separate the df in three sets: training, validation and testing totally random:
        1. The train set contains 80% of molecules from each repository||chromatography condition.
        2. The val set contains 10% of molecules form each repository||chromatography condition.
        3. The test set contains 10% of molecules from each repository||chromatography condition.
A random seed (42) is set, so the random splitting should be reproducible unless this is changed.
The function defined here will create the random split input datafile in: ./data/processed_data/random_split_data/
If the processed datafile does not exist, it will be created from the raw datafile. In the same way, if the raw datafile does not exist, it will be created.
(See ./src/process_RepoRT_data/data_processing.py)
Update (1.1.): Implemented the new version of data_processing.py with the filtered datafile.
Update (1.2.): Adapted to the version 1.2. of data_processing.py, as now this splitting function is more flexible.
"""

# IMPORT MODULES
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.process_RepoRT_data.data_processing import get_processed_df_from_raw

# @todo Decide whether to delete the __main__ block below or update it to pass the
# required arguments explicitly. At line 71 this function is called as
# split_train_val_test() with no arguments, but its signature requires
# input_path, output_dir, drop_smrt, and apply_upthreshold.
def split_train_val_test (input_path, output_dir, drop_smrt, apply_upthreshold):
    """
    Uses the scaled_complete_df to perform train_test_split.
    Random_state set to 42 for consistency.
    """
    file = Path (input_path)
    if file.exists():
        print ("The input file is correct!")
    else:
        print ("Getting the processed datafile...")
        get_processed_df_from_raw(drop_smrt=drop_smrt,
                                  apply_upthreshold=apply_upthreshold,
                                  complete=False)
    df = pd.read_csv(input_path, sep = "\t")

    print ("Checking the output dir...")
    os.makedirs(output_dir, exist_ok = True)

    #Main splitting loop.
    train_list = []
    test_list = []
    val_list = []
    index_array = np.unique (df["dir_id"])
    for index in index_array:
        temp_df = df[df["dir_id"] == index]
        # First split to get the test set (0.1)
        train_df, test_df = train_test_split(temp_df, test_size = 0.1, random_state = 42)
        test_list.append(test_df)
        # Second split to get the train and val set (~0.8 and ~0.1, respectively)
        train_df, val_df = train_test_split(train_df, test_size = 0.1111, random_state = 42)
        train_list.append(train_df)
        val_list.append(val_df)

    print ("Exporting the datafiles...")
    train_set = pd.concat (train_list, ignore_index=True)
    val_set = pd.concat (val_list, ignore_index=True)
    test_set = pd.concat (test_list, ignore_index=True)
    train_filename = output_dir + "train_data.tsv"
    test_filename = output_dir + "test_data.tsv"
    val_filename = output_dir + "val_data.tsv"
    train_set.to_csv (train_filename, sep = "\t", index = False)
    test_set.to_csv (test_filename, sep = "\t", index = False)
    val_set.to_csv (val_filename, sep = "\t", index = False)
    print (f"Successfully split the processed datafile randomly and the datafiles will be saved in {output_dir}")

if __name__ == "__main__":
    split_train_val_test()
    train_file = output_dir + "train_data.tsv"
    val_file = output_dir + "val_data.tsv"
    test_file = output_dir + "test_data.tsv"
    a = pd.read_csv (train_file, sep = "\t")
    b = pd.read_csv (val_file, sep = "\t")
    c = pd.read_csv (test_file, sep = "\t")
