"""
    Strategy:
        1. Starts with the biggest cc or ms sccafold group.
        2. Fill the fold sets with big sets first, then start filling with the smaller ones checking the size.
"""


# IMPORT MODULES

import os

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold

from chemprop import data, featurizers
from rdkit.Chem import MolFromInchi

# PARAMETERS FOR KFOLD

K = 10                                                                          # By default, 5 folds will be made.
ROOT_NAME = "k-fold"
SIZE_DICT = {f"k-fold{fold_index}":0 for fold_index in range(1, K+1)}           # This will store the size of each split
OBJECTIVE_DICT = {f"k-fold{fold_index}":[] for fold_index in range(1, K+1)}     # This will store the cc or murcko scaffold
RANDOM_SEED = 42                                                                # The random seed number for all splittings.

# RANDOM SPLIT K-FOlD FUNCTIONS

def get_random_split_kfolds (df,
                             objective_dict,
                             random_seed=RANDOM_SEED):
    """
        This function splits the input df into 10 similar folds having similar size.
        Accepts a random_seed argument, by default is set 42 to be reproducible.
        The output is an array of DataFrames.
    """

    dir_id_array = np.unique(df["dir_id"])
    for dir_id in dir_id_array:
        temp_df = df [df["dir_id"] == dir_id]
        temp_folds = KFold(n_splits=10, random_state=random_seed, shuffle=True)
        fold_count = 0
        for _, fold_index in temp_folds.split(temp_df):
            fold_count += 1
            fold_df = temp_df.iloc[fold_index]
            objective_dict[f"k-fold{fold_count}"].append(fold_df)

    final_df_array = [ pd.concat(fold_df_array) for fold_df_array in objective_dict.values()]

    return final_df_array

# cc_split and ms_split functions

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




# Model_per_repo functions

def mpr_get_train_loader (train_df):
    """
        This is different to any other similar functions. As it takes a TRAIN DataFrame only.
        Also, the input data (metadata and gradient data) does not need scaling as they will be the same for all molecules.
        Note: "mpr (Model Per Repo)"
    """

    rt_array = train_df.loc [:,["rt"]].values
    inchi_array = train_df.loc[:, "inchi.std"].values

    mols = [ MolFromInchi(inchi) for inchi in inchi_array]
    all_data = [[data.MoleculeDatapoint(mol, rt) for mol, rt in zip(mols, rt_array)]]

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer ()
    train_dset = data.MoleculeDataset(all_data[0], featurizer)
    train_scaler = train_dset.normalize_targets()  # For the targets
    train_loader = data.build_dataloader(train_dset,
                                         num_workers=5,
                                         shuffle=True,
                                         seed=RANDOM_SEED)
    return train_loader, train_scaler

def mpr_get_val_loader (val_df, train_scaler):
    """
        Same as the previous function's description
    """
    rt_array = val_df.loc[:, ["rt"]].values
    inchi_array = val_df.loc[:, "inchi.std"].values

    mols = [MolFromInchi(inchi) for inchi in inchi_array]
    all_data = [[data.MoleculeDatapoint(mol, rt) for mol, rt in zip(mols, rt_array)]]

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    val_dset = data.MoleculeDataset(all_data[0], featurizer)

    val_dset.normalize_targets(train_scaler)  # Here this is normalized with the train_scaler.
    val_loader = data.build_dataloader(val_dset,
                                        num_workers=5,
                                        shuffle=False)
    return val_loader

def mpr_get_test_loader (test_df):
    """
        See description for function "mpr_get_train_loader".
        The targets are not scaled.
    """
    rt_array = test_df.loc[:, ["rt"]].values
    inchi_array = test_df.loc[:, "inchi.std"].values

    mols = [MolFromInchi(inchi) for inchi in inchi_array]
    all_data = [[data.MoleculeDatapoint(mol, rt) for mol, rt in zip(mols, rt_array)]]

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    test_dset = data.MoleculeDataset(all_data[0], featurizer)

    test_loader = data.build_dataloader(test_dset,
                                        num_workers=5,
                                        shuffle=False)
    return test_loader

# Save folds function. NOT APPLIED TO "model_per_repo" as we have no interest in saving ALL FOLDS.

def _write_cc_or_scaffold_report (fold_df,column_name,  path2dir, filename):
    """
        Writes a report file depending on the fold dataframe given, this will be defined by the column name.
        For now, only used for "ms_smiles" or "dir_id".
    """
    unique_values = np.unique (fold_df [column_name])
    unique_size_dict = {unique_value: len(fold_df[fold_df[column_name]==unique_value])
                        for unique_value in unique_values}
    path2reports = os.path.join(path2dir, "report_files/")
    os.makedirs(path2reports, exist_ok=True)

    path2file = os.path.join(path2reports, filename)

    with open(path2file, "w") as f:
        f.write (f"The unique values contained in this fold are:\n")
        f.writelines (f"{unique_value}: {size} molecules\n" for unique_value, size in unique_size_dict.items())
    return

def save_random_split_kfolds (df_array, path2dir):
    """
        This function saves the folds for the random_split k-folds.
        Has no reports to write, as no unique values are found.
    """
    os.makedirs (path2dir, exist_ok=True)

    for index, fold_df in enumerate(df_array):
        path2file = os.path.join (path2dir, f"fold_{index}.tsv")
        fold_df.to_csv (path2file, sep="\t", index=False)

    print ("Finished saving all folds!")

def save_cc_or_scaffold_kfolds (df_array, column_name, path2dir):
    """
        As the previous function, the path2dir is the seed dir to save all 10 folds.
        This save all 10 folds as tsv files in the same directory but with different names.
    """

    os.makedirs(path2dir, exist_ok=True)
    temp_dict = {"fold":[],
                 "size":[]}

    for index, fold_df in enumerate(df_array):
        fold_name = f"fold_{index}"
        temp_dict["fold"].append(fold_name)
        temp_dict["size"].append(len(fold_df))

        fold_filename = os.path.join (path2dir, f"{fold_name}.tsv")
        report_name = f"{fold_name}_report.txt"

        fold_df.to_csv(fold_filename, sep='\t', index=False)
        _write_cc_or_scaffold_report(fold_df, column_name, path2dir, report_name)
    print ("Finished saving all folds!")

    print ("Saving the size report tsv")
    size_df = pd.DataFrame(temp_dict)
    size_df.to_csv (os.path.join (path2dir, "size_report.tsv"),
                    sep='\t',
                    index=False)
    return

# Results functions

def write_overall_results (result_dict, path2output):
    """
        This function summarizes the results from all 10 runs into a single dataframe and saved in a .tsv file.
        Also, another .tsv file is written with the mean and standard deviation for each metric.

    """
    k = len(result_dict["MAE"])
    summarized_df = pd.DataFrame (result_dict, index=([f"run_{i}" for i in range(1, k+1)]))
    temp_metric_dict = {"metric":[],
                        "mean":[],
                        "standard deviation":[]}
    for column in summarized_df.columns:
        temp_metric_dict["metric"].append(column)
        temp_metric_dict["mean"].append(summarized_df[column].mean())
        temp_metric_dict["standard deviation"].append(summarized_df[column].std())
    metric_df = pd.DataFrame(temp_metric_dict)
    path2overall_metrics_file = os.path.join (path2output, "overall_metrics.tsv")
    metric_df.to_csv (path2overall_metrics_file, sep='\t', index=False)

    summarized_df ["test_fold"] = list(range(k))
    summarized_df ["val_fold"] = [(i + 1)%k for i in range(k)]

    path2summirized_file = os.path.join(path2output, "result_summary.tsv")
    summarized_df.to_csv (path2summirized_file, sep='\t')


def _calculate_all_metrics (df):
    """
        This is a helper function to calculate all weighted metrics.
        Returns a dataframe with dir_id and all the weighted metrics.
    """
    weights= df["n_molecules"]
    mean_mae = np.average(df["MAE"], weights= weights)
    std_mae = np.sqrt(np.average((df["MAE"] - mean_mae)**2 , weights=weights))
    mean_rmse = np.average(df["RMSE"], weights= weights)
    std_rmse = np.sqrt(np.average((df["RMSE"] - mean_rmse)**2, weights= weights))
    mean_mre = np.average(df["MRE"], weights= weights)
    std_mre = np.sqrt(np.average((df["MRE"] - mean_mre)**2, weights= weights))
    mean_rel_max_rt_error = np.average(df["rel_max_rt_error"], weights= weights)
    std_rel_max_rt_error = np.sqrt(np.average((df["rel_max_rt_error"] - mean_rel_max_rt_error) **2, weights= weights))
    mean_rel_mean_rt_error = np.average(df["rel_mean_rt_error"], weights= weights)
    std_rel_mean_rt_error = np.sqrt(np.average((df["rel_mean_rt_error"] - mean_rel_mean_rt_error) **2, weights= weights))

    metric_row_df = pd.DataFrame ({"mean_MAE":[mean_mae],
                                  "std_MAE":[std_mae],
                                  "mean_RMSE":[mean_rmse],
                                  "std_RMSE":[std_rmse],
                                  "mean_MRE":[mean_mre],
                                  "std_MRE":[std_mre],
                                  "mean_rel_max_rt_error":[mean_rel_max_rt_error],
                                  "std_rel_max_rt_error":[std_rel_max_rt_error],
                                  "mean_rel_mean_rt_error":[mean_rel_mean_rt_error],
                                  "std_rel_mean_rt_error":[std_rel_mean_rt_error],
                                   })

    return metric_row_df



def write_model_per_repo_results (results_df, path2output):
    """
        Here 2 different metrics are calculated. The mean and std deviation for each dir_id using the metric value from the Folds.
        The aggregated metric (weighted mean and std deviation) using the value calculated from all folds and weighted by the molecule count for each repo.
    """

    print(f"Saving the results in {path2output}...")

    # Save the input df first.

    path2allfolds_table = os.path.join (path2output, "all_folds.tsv")
    results_df.to_csv (path2allfolds_table, sep='\t', index=False)

    # Make the metric_per_cc file

    temp_array = []
    dir_id_array = np.unique (results_df["dir_id"])
    for dir_id in dir_id_array:
        temp_df = results_df[results_df["dir_id"] == dir_id]
        temp_res_df = _calculate_all_metrics(temp_df)
        temp_array.append(temp_res_df)

    metric_per_repo_df = pd.concat(temp_array, ignore_index=True)
    metric_per_repo_df.insert(0, "dir_id", dir_id_array)
    path2metricperrepo_table = os.path.join (path2output, "metric_per_cc.tsv")
    metric_per_repo_df.to_csv(path2metricperrepo_table, sep='\t', index=False)

    # The aggregated metric file.

    all_df = _calculate_all_metrics (results_df)
    path2alldf_table = os.path.join(path2output, "aggregated_results.tsv")
    all_df.to_csv(path2alldf_table, sep='\t', index=False)

    print (f"All metric files are written and saved in {path2output}")

if __name__ == "__main__":
    SIZE_DICT = {f"k-fold{fold_index}": 0 for fold_index in range(1, K + 1)}  # This will store the size of each split
    OBJECTIVE_DICT = {f"k-fold{fold_index}": [] for fold_index in
                      range(1, K + 1)}  # This will store the cc or murcko scaffold
    path2input = os.path.join(".", "data", "RepoRT_RP", "processed_data", "no_SMRT","complete_processed_data.tsv")
    df = pd.read_csv (path2input, sep="\t")
    res_array = split_dataset_into_k_folds(df,  OBJECTIVE_DICT, SIZE_DICT,"dir_id")
    save_cc_or_scaffold_kfolds(res_array,"dir_id","./")

