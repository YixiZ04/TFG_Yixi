"""
Name: data_processing.py
Author: Yixi Zhang
Date: March 2026
Version: 1.2.
Usage: The processing method:
    1. Keep only those columns used for training.
    2. Drop those molecules having > 10 gradients.
    3. Get max and mean RT for each repository, merge them to each molecule according the dir_id.
Only functions are defined here, but if called the function "get_processed_df_from_raw", the processed datafile will be created and saved in:
./data/processed_RepoRT/complete_treated_data.tsv.
(See ./src/get_RepoRT_data/RepoRT_get_all_data.py)
Update: Rectified a concept error here, as the final result table should not be scaled, and the scaling process will be done after data splitting using train set Scaler
Also, on top of the previous update, here more options are given:
    1. Drop completely the SMRT dataset and apply down_threshold. (Drop datasets if contains less molecules than the threshold).
    2. Not dropping completely the SMRT and apply a up_threshold, meaning that we are performing a downsampling process.
Overall, if all options were tested, this file could produce 6 different tsv files in ./data/processed_RepoRT/ and identified by their filename:
    1. no_SMRT_complete_data.tsv (Dropped SMRT but no any filter has been applied, this is, it contains repos < 100 molecules)
    2. no_SMRT_ds_data.tsv (Dropped SMRT and applied downsampling (ds), the repos contains 100 < n < 5000 molecules)
    3. no_SMRT_no_ds_data.tsv (Dropped SMRT and applied ds, the repos contains n > 100 molecules.
    4. with_SMRT_complete_data.tsv (Not dropped SMRT and repos contains any number of molecules).
    5. with_SMRT_ds_data.tsv. This would be the "filtered_treated_data.tsv" in the previous version applying the filters.
    6. with_SMRT_no_ds_data.tsv. Containing SMRT and no ds applied, but all repos contains n > 100 molecules.
NOTE: This is possible to create all 6 files, but it does not mean that we will evaluate models on all of them.
"""
# IMPORT MODULES AND SCRIPTS
import pandas as pd
import numpy as np
import os
from pathlib import Path
from src.get_raw_data.RepoRT_get_all_data import get_raw_datatable, save_dir as raw_save_dir

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

def update_df_drop_grad (df, threshold, only_metadata=False):
    """
    Given a pandas Dataframe and a threshold, which is the number of gradient from where to drop the gradient data.
    Output: An updated dataframe with the molecules from those repos whose gradient data is > threshold and the columns containing that information dropped.
    Update: A boolean only_metadata; this is added for the script get_project_metadata.py; it has been set to False as default value, so for other usages,
    no changes have to be made.
    """
    drop_index_array = []
    fr_threshold = f"flow rate [ml/min]_{threshold}"
    index_array = np.unique (df["dir_id"])
    for index in index_array:
        temp_df = df[df["dir_id"] == index]
        # The following works because for a repository, only 1 flowrate is found, so if it is not 0, meaning that has the gradient information for the threshold.
        if np.unique(temp_df[fr_threshold]) [0] != 0:
            # print ("found")
            drop_index_array.append (index)
        else:
            continue
    if only_metadata:
        return drop_index_array, threshold
    new_temp_df = df[~df["dir_id"].isin(drop_index_array)] #Here, a treated df is built dropping those repositories having > threshold gradient.
    # Drop those columns containing those data.
    position = new_temp_df.columns.get_loc("flow rate [ml/min]_10") + 1
    updated_df = new_temp_df.iloc[:, :position]
    return updated_df

def get_max_mean_rt_per_cc (complete_df):
    """
    This get the max and mean rt for every chromatography column and inserts them next to "rt_s" column of the dataframe.
    Returns the updated df.
    """
    max_array = []
    mean_array = []
    index_array = np.unique (complete_df["dir_id"])
    for index in index_array:
        temp_df = complete_df [complete_df ["dir_id"] == index]
        mean_rt = np.mean (temp_df["rt_s"])
        max_rt = np.max (temp_df["rt_s"])
        temp_max_array = [ max_rt for _ in range (temp_df.shape[0])]
        temp_mean_array = [ mean_rt for _ in range (temp_df.shape [0])]
        max_array = max_array + temp_max_array
        mean_array = mean_array + temp_mean_array

    position = complete_df.columns.get_loc ("rt_s")
    complete_df.insert (position + 1, "max_rt", max_array)
    complete_df.insert (position + 2, "mean_rt", mean_array)

def filter_by_n_mol (df, down_threshold = 100, up_threshold = 5000, apply_upthreshold = False, only_metadata=False):
    """
    Input: A threshold for filtering RepoRT data. Those repositories containing less than threshold number of molecule will be removed.
    If the cc has more molecules than up_threshold, a downsampling will be applied to that repo and retains the up_threshold number of molecules.
    apply_upthreshold (False): Boolean to decide whether to perform downsampling or not.
    Output: The updated dataframe containing the filtered data.
    Update: A boolean only_metadata; this is added for the script get_project_metadata.py; it has been set to False as default value, so for other usages,
    no changes have to be made.
    """
    index_array = np.unique (df["dir_id"])
    final_df, dropped_repo, ds_repo = [], [], []
    if apply_upthreshold:
        for index in index_array :
            temp_df = df[df["dir_id"] == index]
            if down_threshold < temp_df.shape[0] < up_threshold:
                final_df.append (temp_df)
            elif temp_df.shape[0] > up_threshold:
                final_df.append (temp_df.sample (up_threshold, random_state=42))
                ds_repo.append (index)
            else:
                dropped_repo.append (index)
                continue
    else:
        for index in index_array:
            temp_df = df[df["dir_id"] == index]
            if down_threshold < temp_df.shape[0]:
                final_df.append(temp_df)
            else:
                dropped_repo.append (index)
                continue
    if only_metadata:
        return dropped_repo, ds_repo
    return pd.concat (final_df)



def get_processed_df_from_raw (input_path = "./data/no_extra_mol_desc/RepoRT_complete_data.tsv",
                               path2res = "./data/processed_RepoRT/",
                               grad_num2drop = 11,
                               down_threshold = 100,
                               up_threshold = 5000,
                               complete = False,
                               drop_smrt = True,
                               apply_upthreshold = False,
                               smrt_id = 186,
                               ):
    """
    Input: complete (Boolean), default set to False to get the filtered data; if set to true, the complete data without filtering is fetched.
    input_path to RepoRT raw data.
    path2res, the directory for saving the result files.
    grad_num2drop, the number to drop from the gradient dataframe.
    down_threshold and up_threshold: parameters used for filtering; up_threshold mainly for downsampling.
    Output: Depending on how "complete" Boolean is set, the result file will be different:
        *False: ./data/processed_RepoRT/filtered_treated_data.tsv
        *True: ./data/processed_RepoRT/complete_treated_data.tsv
    Update: Takes 2 new Booleans and has 1 default parameter:
        *drop_smrt (True): True if want to drop completely the SMRT dataset.
        *apply_upthreshold (False): True if want to perform downsampling.
        *smrt_id (186): This is the dir_id of SMRT, only change if RepoRT reorganized the directory IDs, which is not very likely
    """
    print (f"Checking for input file...")
    file = Path (input_path)
    source_path = input_path
    if file.exists():
        print (f"Input file read successfully!")
    else:
        print (f"Input file not found, creating new file...")
        get_raw_datatable()
        source_path = os.path.join(raw_save_dir, "RepoRT_complete_data.tsv")
        print (f"Reading regenerated raw data from default path: {source_path}")
    df = pd.read_csv(source_path, sep = "\t")
    if drop_smrt: #Not only decides if drop SMRT at all, but also initialize the filename for saving.
        print ("Dropping SMRT data...")
        filename = path2res + "no_SMRT_"
        df = df[df["dir_id"] != smrt_id]
    else:
        filename = path2res + "with_SMRT_"

    print ("Checking for saving directory...")
    os.makedirs(path2res, exist_ok = True)

    # The whole process
    print ("Starting processing the data...")
    temp_df = keep_only_useful_columns(df)
    temp_df = update_df_drop_grad(temp_df, grad_num2drop)
    get_max_mean_rt_per_cc(temp_df)
    if complete:
        filename = filename + "complete_data.tsv"
        print(f"The complete df is built and it will be saved in {filename}")
        temp_df.to_csv (filename, sep = '\t', index = False)
        print(f"Complete saving the precessed file in {filename} !!")
        return
    else:
        final_filtered_df = filter_by_n_mol(temp_df, down_threshold = down_threshold, up_threshold = up_threshold, apply_upthreshold = apply_upthreshold)
        if apply_upthreshold:
            filename = filename + "ds_data.tsv" #Note: ds stands for "downsampling"
        else:
            filename = filename + "no_ds_data.tsv"
        print(f"The filtered df is built and it will be saved in {filename}")
        final_filtered_df.to_csv(filename, sep = '\t', index = False)
        print(f"Complete saving the precessed file in {filename} !!")
        return

if __name__ == "__main__":
    get_processed_df_from_raw()
