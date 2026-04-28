"""
Name: RepoRT_MPNN_each_repo.py
Author: Yixi Zhang
Date: March 2026/April 2026
Version: 1.2.
Usege: Builds individuals MPNN using data from every RepoRT's subsets (0001, 0002...
In total, there will be 3 main result files saved in the result directory (should be defined):
    1. metric.txt. This file contains the aggregated metrics (MAE, RMSE, %errors) calculated from Results.tsv. Might not be as interesting as the last result file,
    but it is interesting if wanted to have a quick idea of the difference of the results.
    2. Parameters.txt: a txt file containing the parameters used for building the MPNN. It is shared by all repositories.
    3. Results.tsv: a tsv file containing the prediction results for each subset of RepoRT.
    4. metrics_per_cc.tsv. This is the original "Metrics.tsv", but this now contains more results metrics (%error to mean and max RT).
Update: This will use the total processed RepoRT dataset for making sure that the repos used in this Script corresponds with the repos remaining after processing.
With this update, very similar Scripts will be removed and the results will be comparable to those obtained with models trained using all data.
The result files are the exact same as the result files for other cases (random_split e.g.).The only difference being on how those results are obtained.
Update (1.2): added option to train models with moldescs
"""

# IMPORT THE FUNCTIONS NEEDED

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from chemprop import  nn
from lightning import pytorch as pl
from src.training.functions.basic_model_functions import get_dataloaders, configure_and_train_mpnn
from src.training.functions.moldesc_model_functions import get_dataloaders_with_moldesc, configure_and_train_mpnn_moldesc
from src.training.functions.splitted_sets_functions import *
from src.RepoRT_data_processing.RepoRT_processing import get_processed_df_from_raw

# DEFINE THE PARAMETERS. HERE DEFINED ARE THE DEFAULT VALUES OF CHEMPROP
SOURCE_PATH = os.path.join(".", "data", "RepoRT_RP", "processed_data/")                        # This is the source directory that contains all processed files
dataset_type = "no_SMRT"                                                                  # Or with_SMRT, depends on the type of input dataset to use.
apply_grad_down_threshold = False                                                           # Set to True if want to use the filtered by grad_down_threshold
filtering = "filtered" if apply_grad_down_threshold else "no_filtered"
using_moldescs = False                                                                      # Set to True if want to use molecular descriptors for the model
moldesc_dir = "RepoRT_moldesc" if using_moldescs else "RepoRT_RP"                              # Changes the path where to save the results files
path2res = os.path.join(".", "logs", moldesc_dir, dataset_type, filtering, "model_per_repo", "01_20_04_2026/") #Change "dirname" for any name you want.
path2moldesc = os.path.join (".", "data","complete_moldesc.tsv")



param_dict = {
    "mp_hidden_dim": 451,                             # Hidden dimension of the message passing (MP) part
    "mp_depth": 4,                                    # Depth/Number of Layers of the MP
    "ffn_hidden_dim": 1493,                            # Hidden layer for the feed-forward network (ffn). This is the regressor
    "ffn_layers": 4,                                  # Number of layers for the ffn.
    "init_lr": 1e-4,                                  # The initial learning rate (lr)
    "max_lr": 1e-3,                                   # Max lr will be reached in after the warm_up epochs.
    "final_lr": 1e-4,                                 # The lr set for the rest of epochs.
    "warm_up_epochs": 2,                              # Number of epochs to reach the max_lr
    "max_epochs": 1000,                               # Set to a smaller number as the datasets here are much smaller.
    "dropout_rate": 0.1,                              # Dropout rate. 0 is default.
    "batch_norm": True,                               # True if want to apply batch_norm
    "metric_list": [nn.MAE(), nn.RMSE()],
    "accelerator": "auto",                            # If GPU and CUDA available change to "gpu". Or can set "cpu" as well.
}




# RUNNING THE SCRIPT
if __name__ == "__main__":
    pl.seed_everything(42, workers=True)

    # Assertion for the data type. Match is used here for better generalization if in the future more dataset types will be evaluated.
    match dataset_type:
        case "no_SMRT":  # These following Booleans are used to get the processed dataset if has not been created yet.
            drop_smrt = True
            if apply_grad_down_threshold:
                path2input = os.path.join(SOURCE_PATH, "no_SMRT_down_grad_filter/")
            else:
                path2input = os.path.join(SOURCE_PATH, "no_SMRT/")
        case "with_SMRT":
            drop_smrt = False
            if apply_grad_down_threshold:
                path2input = os.path.join(SOURCE_PATH, "with_SMRT_down_grad_filter/")
            else:
                path2input = os.path.join(SOURCE_PATH, "with_SMRT/")
        case _:
            raise NameError(f"Check the dataset_type: {dataset_type}.")

    input_file = os.path.join (path2input, "complete_processed_data.tsv")

    # Checking fot the input file existence

    if not Path(input_file).exists():
        print (f"Making the input file: {input_file}...")
        get_processed_df_from_raw(drop_smrt=drop_smrt,
                                  down_grad_filter=apply_grad_down_threshold,)
    print ("Getting the input df...")
    df = pd.read_csv (input_file, sep="\t")

    # Make sure the directory exists.
    print ("Checking for the output directory...")
    os.makedirs(path2res, exist_ok=True)

    #Training process.
    dir_id_array = np.unique (df["dir_id"])
    results_array = []          # This array will be used store dfs to build a large df for results, where all the metrics will be calculated.
    for dir_id in dir_id_array:
        temp_df = df[df["dir_id"] == dir_id]        # This id can be directly used because when imported from tsv file, those "0"s would be eliminated.
        # temp_df = temp_df.sample (50)         #Run this to have a quick test of the Script's usage

        # Build a model for each repo.
        if using_moldescs:
            temp_df = add_moldescs(temp_df, path2moldesc)
            targets_scaler, mol_descs_scaler, train_loader, val_loader, test_loader, test_indices = get_dataloaders_with_moldesc(temp_df,
                                                                                                       dataset="RepoRT")
            mpnn, trainer = configure_and_train_mpnn_moldesc(targets_scaler, mol_descs_scaler, train_loader, val_loader, param_dict, path2res)
        else:
            inchis_array = temp_df.loc[:, "inchi.std"].values
            rts = temp_df.loc[:, ["rt"]].values
            scaler, train_loader, val_loader, test_loader, test_indices = get_dataloaders(feature_array=inchis_array,
                                                                                          target_array=rts)
            mpnn, trainer = configure_and_train_mpnn(scaler, train_loader, val_loader, param_dict, path2res, save_model=False)

        test_pred = trainer.predict(mpnn, test_loader)
        test_pred = np.concatenate(test_pred, axis=0)
        # GETTING RESULTS
        print (f"Getting results for the repo {dir_id}")
        temp_test_df = temp_df.iloc [test_indices[0]]
        temp_res_table = get_res_table(temp_test_df, test_pred, path2res, save_results=False, using_moldescs=using_moldescs)
        results_array.append(temp_res_table)
        del mpnn, trainer
    print (f"Writting the final result files in {path2res}...")
    res_table = pd.concat (results_array, ignore_index=True)
    res_table.to_csv (path2res+ "Results.tsv", sep = "\t", index = False)
    mae, rmse, mre,rel_max_error, rel_mean_error = metrics_from_dataframe(res_table)
    write_parameters_file(param_dict, path2res)
    write_metrics_per_cc(res_table, path2res)
    write_metric_txt(mae, rmse, mre,rel_max_error, rel_mean_error, path2res)

    print ("The resuls written successfully! Exiting the program...")
    sys.exit(0)


