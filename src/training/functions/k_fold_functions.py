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

def save_kfold (df_array, path2dir):
    k = len(df_array)

    for i in range(k):
        test_df = df_array[i]
        val_df =df_array[(i + 1) % k]
        train_df = [
            df_array[j]
            for j in range(k)
            if j != i and j != (i + 1) % k
        ]
        train_df = pd.concat(train_df, ignore_index=True)
        path2save = os.path.join(path2dir, f"k_fold_{i}/")
        os.makedirs(path2save, exist_ok=True)
        test_df.to_csv(os.path.join(path2save, "test.tsv"), sep='\t', index=False)
        train_df.to_csv(os.path.join(path2save, "train.tsv"), sep='\t', index=False)
        val_df.to_csv(os.path.join(path2save, "val.tsv"), sep='\t', index=False)
        report_filename = os.path.join(path2save, "report.txt")
        with open(report_filename, "w") as f:
            f.write("The train set contains:\n")
            f.writelines(f"{dir_id}\n" for dir_id in np.unique(train_df["dir_id"]))
            f.write("The val set contains:\n")
            f.writelines(f"{dir_id}\n" for dir_id in np.unique(val_df["dir_id"]))
            f.write("The test set contains:\n")
            f.writelines(f"{dir_id}\n" for dir_id in np.unique(test_df["dir_id"]))

# Results functions


if __name__ == "__main__":
    SIZE_DICT = {f"k-fold{fold_index}": 0 for fold_index in range(1, K + 1)}  # This will store the size of each split
    OBJECTIVE_DICT = {f"k-fold{fold_index}": [] for fold_index in
                      range(1, K + 1)}  # This will store the cc or murcko scaffold
    path2input = os.path.join(".", "data", "RepoRT", "processed_data","no_SMRT", "complete_processed_data.tsv")
    input_df = pd.read_csv(path2input, sep="\t")
    final_array = split_dataset_into_k_folds(input_df, OBJECTIVE_DICT,SIZE_DICT, "dir_id")
    path2dir = os.path.join (".", "data", "RepoRT", "processed_data", "no_SMRT", "kfolds/")
    os.makedirs(path2dir, exist_ok=True)
    save_kfold(final_array, path2dir)

