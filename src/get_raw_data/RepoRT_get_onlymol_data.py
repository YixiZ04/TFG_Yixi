"""
Name: get_onlymol_data_RepoRT.py
Author: Yixi Zhang
Date: March 2026
Version: 1.0.
Usage: This script get only the processed retention time data for RepoRT from the url:
https://raw.githubusercontent.com/michaelwitting/RepoRT/refs/heads/master/processed_data/{repo_index}/{repo_index}_rtdata_canonical_success.tsv
https://raw.githubusercontent.com/michaelwitting/RepoRT/refs/heads/master/processed_data/{repo_index}/{repo_index}_rtdata_isomeric_success.tsv
This data could be used for training a model for each repo from RepoRT and compare the result with the model trained with the entire RepoRT (including
the cromatography conditions)-
Some preprocessing will be done during the process:
    1. Those repositories contaningn less than 50 moleucles will be dropped.
    2. An additional column containing the RT data in seconds will be added.
"""


# Import modules

import pandas as pd
import numpy as np
import os
import sys

# Define some variables.

seed_url = "https://raw.githubusercontent.com/michaelwitting/RepoRT/refs/heads/master/processed_data/"
num_repos = 440                         # As right now, 0438 is the max index of the repos.
path2dir = "./data/no_extra_mol_desc/"
os.makedirs(path2dir, exist_ok=True)            # Make the directory if not existing

# Define functions for data importing
def get_index_array (num_repos):
    """
    Input: The number of repos you want to use for training
    Output: a numpy array containing the indexes string in the specific format ("0001", e.g.)
    """
    index_array = []
    for index in range(1, num_repos+1):
        index = str (index)
        while len (index) < 4:
            index = "0" + index
        index_array.append (str (index))
    return np.array(index_array)
def get_repoRT_index (index):
    """
    From a index (1, 2, 3...) get the RepoRT index ("0001", "0002", "0003"...)
    """
    index = str (index)
    while len(index) < 4:
        index = "0" + index
    return index
def get_rt_data (num_repos=num_repos,seed_url=seed_url):
    """
    Input: num_repos to fetch, and the seed url for RepoRT.
    Output: a table containing the data fetched from RepoRT
    """
    final_dataframe = pd.DataFrame()
    index_array = get_index_array(num_repos)
    for index in index_array:
        can_url = f"{seed_url}{index}/{index}_rtdata_canonical_success.tsv"
        iso_url = f"{seed_url}{index}/{index}_rtdata_isomeric_success.tsv"
        print (f"Fetching RT data for nº{index}...")
        try:
            temp_dataframe_can = pd.read_csv(can_url, sep="\t", encoding="utf-8")
            temp_dataframe_iso = pd.read_csv(iso_url, sep="\t", encoding = "utf-8")
            total_mol_count = temp_dataframe_can.shape [0] + temp_dataframe_iso.shape [0]
            if total_mol_count > 50 and temp_dataframe_iso.shape [0] != 0:
                final_dataframe = pd.concat ([final_dataframe, temp_dataframe_can, temp_dataframe_iso], ignore_index=True)
            elif total_mol_count > 50 and temp_dataframe_iso.shape [0] == 0:  #If not isomeric_success.tsv is empty (0 rows)
                final_dataframe = pd.concat ([final_dataframe, temp_dataframe_can], ignore_index=True)
            else:
                print (f"The repo {index} contains less than 50 molecules. It will be skipped.")
        except:
            print (f"The repo nº {index} has not been found in the dataset. It will be skipped...")
    dir_id_array = [idmol.split("_")[0] for idmol in final_dataframe["id"]]
    dir_id_array = np.array(dir_id_array)
    final_dataframe = final_dataframe.rename(columns={"id":"molecule_id"})
    final_dataframe.insert(0, "dir_id", dir_id_array)

    #Insert the rt_s column
    position = final_dataframe.columns.get_loc ("rt") + 1 #Just next to the "rt" (minutes) column
    rt_s_array =final_dataframe ["rt"] * 60
    final_dataframe.insert (position, "rt_s", rt_s_array)
    final_dataframe ["rt_s"] = round (final_dataframe["rt_s"],2)

    #Save the file
    del temp_dataframe_can, temp_dataframe_iso, dir_id_array
    return final_dataframe

def get_max_mean_rt_per_cc (complete_df):
    """
    This get the max and mean rt for every chromatography column and inserts them next to "rt_s" column of the dataframe.
    Returns the updated df.
    """
    max_array = []
    mean_array = []
    for index in range (1, 440):
        temp_df = complete_df [complete_df ["dir_id"] == get_repoRT_index(index)]
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

def get_final_dataframe (path2dir = path2dir):
    temp_df = get_rt_data()
    final_df = get_max_mean_rt_per_cc(temp_df)
    filename = path2dir + "RepoRT_only_mol_data.tsv"
    final_df.to_csv(filename, sep="\t", index=False)


