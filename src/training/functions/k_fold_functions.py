"""
    Strategy:
        1. Starts with the biggest cc or ms sccafold group.
        2. Fill the fold sets with big sets first, then start filling with the smaller ones checking the size.
"""



import os
import pandas as pd
import numpy as np

K = 10                                                                           # By default, 5 folds will be made.
ROOT_NAME = "k-fold"
SIZE_DICT = {f"k-fold{fold_index}":0 for fold_index in range(1, K+1)}           # This will store the size of each split
OBJECTIVE_DICT = {f"k-fold{fold_index}":[] for fold_index in range(1, K+1)}         # This will store the cc or murcko scaffold

# Functions

def _get_subset_sizes (df, column_name):
    """
    Given a pandas dataframe and the column name, returns another pandas dataframe containing the subset indexes as keys and the size of the subset as values.
    The Output dictionary will be sorted by size.
    """
    subset_name_array = np.unique(df[column_name])
    subset_size_array = []
    for subset_name in subset_name_array:
        temp_df = df[df[column_name] == subset_name]
        subset_size_array.append(len(temp_df))
    final_df = pd.DataFrame({column_name: subset_name_array,
                             "size": subset_size_array}).sort_values(by=["size"], ascending=False)
    return final_df

def _get_sorted_fold_name(index_size_dictionary):
    """
        Given a dictonary with sizes as values, this function will output the sorted array of keys according to the sizes.
    """
    sorted_dict = sorted(index_size_dictionary.items(), key=lambda item:item[1])
    return list(sorted_dict)

def _get_indexes2split (subset_size_df, objective_dict, size_dict, column_name):
    """
        k is the number of folds to build.
        This function bults a dictionary with index arrays as keys and sizes as values.
    """
    name_array = []
    for index,row in subset_size_df.iterrows():
        subset_size = row ["size"]
        objective = row [column_name]
        smallest_fold = _get_sorted_fold_name(size_dict) [0][0]
        size_dict[smallest_fold] += subset_size
        name_array.append(smallest_fold)
        objective_dict[smallest_fold].append(objective)
    return objective_dict, size_dict

def split_dataset_into_k_folds(df, objective_dict, size_dict, column_name):
    """
        This function will split the dataset into k-fold datasets.
        The output will be an array of DataFrames.
    """
    subset_size_dict = _get_subset_sizes(df, column_name)
    objective_dict, size_dict = _get_indexes2split(subset_size_dict, objective_dict, size_dict, column_name)
    final_array = []
    for index_array in objective_dict.values():
        temp_df = df[df[column_name].isin(index_array)]
        final_array.append(temp_df)
    return final_array

# Results functions


if __name__ == "__main__":
    test_df = pd.read_csv(os.path.join(".", "data", "processed_RepoRT", "with_SMRT", "scaffold_split_data", "ms_complete_data.tsv"), sep='\t')
    final_array = split_dataset_into_k_folds(test_df, OBJECTIVE_DICT,SIZE_DICT, "ms_smiles")
    for df in final_array:
        print (len(df))
