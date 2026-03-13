"""
Name: data_processing.py
Author: Yixi Zhang
Date: March 2026
Version: 1.0.
Usage: The processing method:
        1. Keep only those columns used for training.
        2. Drop those molecules having > 10 gradients.
        3. Normalize values that need normalization, e.g., the column.length, column.ta, column.fr... using StandardScaler.
Only functions are defined here, but if called the function "get_processed_df_from_raw", the processed datafile will be created and saved in:
./data/processed_RepoRT/complete_treated_data.tsv.
(See ./src/get_RepoRT_data/RepoRT_get_all_data.py)
"""
# IMPORT MODULES AND SCRIPTS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.get_raw_data.RepoRT_get_all_data import get_raw_datatable
import os
from pathlib import Path
# DEFINE  FUNCTIONS
def keep_only_useful_columns (df):
    """
    Input: raw df from RepoRT.
    Output: Updated df with only useful columns, dropped columns:
    ["comment", "name", "rt", "inchikey.std", "id", "column.name", "column.ups.code" and all columns containing "classyfire."]
    """
    columns_to_drop = ["comment", "name", "rt", "inchikey.std", "id", "column.name", "column.usp.code"]
    for column in df.columns:
        if "classyfire" in column:
            columns_to_drop.append(column)
    df = df.drop (columns = columns_to_drop)
    return df

def update_df_drop_grad (df, threshold):
    """
    Given a pandas Dataframe and a threshold, which is the number of gradient from where to drop the gradient data.
    Output: An updated dataframe with the molecules from those repos whose gradient data is > threshold and the columns containing that information dropped.
    """
    drop_index_array = []
    fr_threshold = f"flow rate [ml/min]_{threshold}"
    for index in range (1, 440):
        temp_df = df[df["dir_id"] == index]
        if temp_df.shape[0] != 0:
            # The following works because for a repository, only 1 flowrate is found, so if it is not 0, meaning that has the gradient information for the threshold.
            if np.unique(temp_df[fr_threshold]) [0] != 0:
                # print ("found")
                drop_index_array.append (index)
            else:
                continue
        else:
            continue
    new_temp_df = df[~df["dir_id"].isin(drop_index_array)] #Here, a treated df is built dropping those repositories having > threshold gradient.
    # Drop those columns containing those data.
    position = new_temp_df.columns.get_loc("flow rate [ml/min]_10") + 1
    updated_df = new_temp_df.iloc[:, :position]
    return updated_df


def get_updated_meta_grad_data (df):
    """
    This is used for getting updated gradient and column metadata from a treated dataframe.
    Get the first row from each index existing in df.
    """
    temp_list = []
    for i in range (1,440):
        temp_df = df[df["dir_id"] == i]
        if temp_df.shape[0] != 0: #It is possible to get a empty df, so this is important.
            row = temp_df.iloc[:1]
            temp_list.append (row)
        else:
            continue
    return pd.concat(temp_list)

def scaling_meta_grad_data (updated_df):
    """
    Requieres a df that only contains 1 metadata and gradient data per repo.
    Returns the same dataframe but scaled.
    """
    temp_df = updated_df.loc[:, "column.length":].astype(float)
    std_scaler = StandardScaler()
    scaled_array = std_scaler.fit_transform(temp_df)
    for column, array in zip (temp_df.columns, scaled_array.transpose()):
        temp_df[column] = array
    temp_df.insert(0, "dir_id", updated_df["dir_id"])
    return temp_df

def get_complete_scaled_df (df, updated_df):
    """
    Use the previous unscaled dataframe and the scaled metadata and gradient dataframe.
    And merges them together to get the complete scaled dataframe.
    """
    temp_df = df.loc[:, :"column.usp.code_nan"]
    final_df = pd.merge (temp_df, updated_df, on = "dir_id", how = "inner")
    return final_df

def get_max_mean_rt_per_cc (complete_df):
    """
    This get the max and mean rt for every chromatography column and inserts them next to "rt_s" column of the dataframe.
    Returns the updated df.
    """
    max_array = []
    mean_array = []
    for index in range (1, 440):
        temp_df = complete_df [complete_df ["dir_id"] == index]
        if temp_df.shape[0] != 0:
            mean_rt = np.mean (temp_df["rt_s"])
            max_rt = np.max (temp_df["rt_s"])
            temp_max_array = [ max_rt for _ in range (temp_df.shape[0])]
            temp_mean_array = [ mean_rt for _ in range (temp_df.shape [0])]
            max_array = max_array + temp_max_array
            mean_array = mean_array + temp_mean_array
        else:
            continue
    position = complete_df.columns.get_loc ("rt_s")
    complete_df.insert (position + 1, "max_rt", max_array)
    complete_df.insert (position + 2, "mean_rt", mean_array)
    return complete_df





def get_processed_df_from_raw (input_path = "./data/no_extra_mol_desc/RepoRT_complete_data.tsv", path2res = "./data/processed_RepoRT/",grad_num2drop = 11):
    print (f"Checking for input file...")
    file = Path (input_path)
    if file.exists():
        print (f"Input file read successfully!")
    else:
        print (f"Input file not found, creating new file...")
        get_raw_datatable()
    df = pd.read_csv(input_path, sep = "\t")


    print ("Checking for saving directory...")
    os.makedirs(path2res, exist_ok = True)

    # The whole process
    print ("Starting processing the data...")
    temp_df = keep_only_useful_columns(df)
    temp_df = update_df_drop_grad(temp_df, grad_num2drop)
    updated_meta_grad_data = get_updated_meta_grad_data(temp_df)
    scaled_meta_grad_data = scaling_meta_grad_data(updated_meta_grad_data)
    final_complete_df = get_complete_scaled_df(temp_df, scaled_meta_grad_data)
    final_complete_df = get_max_mean_rt_per_cc(final_complete_df)
    filename = path2res + "complete_treated_data.tsv"
    print (f"The complete df is built and it will be saved in {filename}")
    final_complete_df.to_csv(filename, sep = '\t', index = False)
    print (f"Complete saving the precessed file in {filename} !!")

