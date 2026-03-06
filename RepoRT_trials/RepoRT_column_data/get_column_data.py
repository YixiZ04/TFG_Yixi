import pandas as pd
import numpy as np
import os


    seed_url_metadata ="https://raw.githubusercontent.com/michaelwitting/RepoRT/refs/heads/master/processed_data/"
save_filename = "C:/Users/leonz/Github_reps/TFG_git/RepoRT_trials/RepoRT_column_data/RepoRT_column_data.tsv"
# 1. Fetch column data from RepoRT

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

def fetch_column_data (num_repos, seed_url=seed_url_metadata, save_dir=save_filename):
    index_array = get_index_array (num_repos)
    final_df = pd.DataFrame()
    for index in index_array:
        filename = f"{seed_url}{index}/{index}_metadata.tsv"
        try:
            temp_df = pd.read_csv(filename, sep="\t", encoding= "utf-8")
            final_df = pd.concat ([final_df, temp_df], ignore_index=True)
            print (f"Successfully loaded repo nº {index}")
        except:
            print (f"The repo nº {index} was not found in the dataset. And it is gonna be skipped...")
    del temp_df
    final_df.to_csv (save_dir, sep="\t", index=False)
    return final_df

fetch_column_data(440)

# Load data

df = pd.read_csv (save_filename, sep="\t")

for column in df.columns:
    if len (np.unique (df [column])) == 0:
        print ("This column has no diff")
    else:
        print (f"{column} has diff")
        continue