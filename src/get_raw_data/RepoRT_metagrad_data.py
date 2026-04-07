"""
Name: RepoRT_metagrad_data.py
Author: Yixi Zhang
Date: April 2026
Version: 1.0.
Description: Run this file to get the column metadata and gradient data. In total 3 output .tsv files will be written:
    1. gradient_data.tsv. Containing the gradient data only. But all the NA values are kept.
    2. metadata.tsv. Containing the column metaadata without any imputation.
    3. meta_grad_data.tsv. This file is obtained by a inner join of the 2 previous files.
By defect, these files will be saved in ./data/meta_grad_data/ .
"""
import os
import sys
import pandas as pd
from src.get_raw_data.RepoRT_get_all_data import get_column_metadata, get_gradient_data

# Define important variables

path2res = os.path.join (".", "data", "meta_grad_data/") # This can be customized if preferred.
project_metadata_dir = os.path.join (".", "data", "project_metadata/") # Same as above.

if __name__ == "__main__":
    print ("Checking for the output directory...")
    os.makedirs(path2res, exist_ok=True)
    os.makedirs(project_metadata_dir, exist_ok=True)

    print ("Getting the metadata and gradient data...")
    metadata_df = get_column_metadata()
    graddata_df = get_gradient_data(fillna=False)
    final_df = pd.merge (metadata_df, graddata_df, on = "dir_id", how = "inner") # Inner join both dfs

    print (f"Saving the result tsv file in {path2res}...")
    metadata_df.to_csv (path2res + "metadata.tsv", sep="\t", index=False)
    graddata_df.to_csv (path2res + "gradient_data.tsv", sep="\t", index=False)
    final_df.to_csv (path2res + "meta_grad_data.tsv", index = False, sep = "\t")

    print ("Successfully written all the result tables! Exiting with code 0...")
    sys.exit (0)