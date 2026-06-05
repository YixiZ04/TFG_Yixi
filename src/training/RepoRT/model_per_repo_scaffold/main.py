"""
    Name: model_per_repo_scaffold.main
    Author: Yixi Zhang
    The splitting type within each repository is by Bemis-Murcko scaffold. Which implies that in the train-test-val, each will contain different molecule class.
"""

# IMPORT THE FUNCTIONS NEEDED

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from chemprop import nn
from lightning import pytorch as pl
from src.training.functions.basic_model_functions import get_dataloaders, configure_and_train_mpnn
from src.training.functions.moldesc_model_functions import get_dataloaders_with_moldesc, configure_and_train_mpnn_moldesc
from src.training.functions.splitted_sets_functions import *
from src.RepoRT_data_processing.RepoRT_processing import get_processed_df_from_raw

# DEFINE THE PARAMETERS. HERE DEFINED ARE THE DEFAULT VALUES OF CHEMPROP
SOURCE_PATH = os.path.join(".", "data", "RepoRT_RP", "processed_data/")                        # This is the source directory that contains all processed files
dataset_type = "with_SMRT"                                                                  # Or with_SMRT, depends on the type of input dataset to use.
apply_grad_down_threshold = False                                                           # Set to True if want to use the filtered by grad_down_threshold
filtering = "filtered" if apply_grad_down_threshold else "no_filtered"
using_moldescs = True                                                                      # Set to True if want to use molecular descriptors for the model
moldesc_dir = "RepoRT_moldesc" if using_moldescs else "RepoRT_RP"                              # Changes the path where to save the results files
path2res = os.path.join(".", "logs", moldesc_dir, dataset_type, filtering, "model_per_repo_scaffold", "01_06_2026/") #Change "dirname" for any name you want.
path2moldesc = os.path.join (".", "data","complete_moldesc.tsv")



param_dict = {
    "mp_hidden_dim": 460,                             # Hidden dimension of the message passing (MP) part
    "mp_depth": 4,                                    # Depth/Number of Layers of the MP
    "ffn_hidden_dim": 1400,                            # Hidden layer for the feed-forward network (ffn). This is the regressor
    "ffn_layers": 3,                                  # Number of layers for the ffn.
    "init_lr": 1e-4,                                  # The initial learning rate (lr)
    "max_lr": 1e-3,                                   # Max lr will be reached in after the warm_up epochs.
    "final_lr": 1e-4,                                 # The lr set for the rest of epochs.
    "warm_up_epochs": 2,                              # Number of epochs to reach the max_lr
    "max_epochs": 1000,                               # Set to a smaller number as the datasets here are much smaller.
    "dropout_rate": 0.12,                              # Dropout rate. 0 is default.
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
    cant_train_array = []
    #Training process.
    cc_id_array = np.unique (df["cc_id"])
    results_array = []          # This array will be used store dfs to build a large df for results, where all the metrics will be calculated.
    for cc_id in ["cc_0", "cc_1"]:
        temp_df = df[df["cc_id"] == cc_id]        # This id can be directly used because when imported from tsv file, those "0"s would be eliminated.

        # Build a model for each repo.
        if using_moldescs:
            try:
                temp_df = add_moldescs(temp_df, path2moldesc)
                targets_scaler, mol_descs_scaler, train_loader, val_loader, test_loader, test_indices = get_dataloaders_with_moldesc(temp_df,
                                                                                                           dataset="RepoRT")
                mpnn, trainer = configure_and_train_mpnn_moldesc(targets_scaler, mol_descs_scaler, train_loader, val_loader, param_dict, path2res)
                test_pred = trainer.predict(mpnn, test_loader)
                test_pred = np.concatenate(test_pred, axis=0)
                # GETTING RESULTS
                print(f"Getting results for the repo {cc_id}")
                temp_test_df = temp_df.iloc[test_indices[0]]
                temp_res_table = get_res_table(temp_test_df,
                                               test_pred,
                                               path2res,
                                               save_results=False,
                                               using_moldescs=using_moldescs)
                results_array.append(temp_res_table)
                del mpnn, trainer
            except:
                print(f"{cc_id} does not contain > 3 Bermis-Murcko scaffolds")
                cant_train_array.append(cc_id)
                continue
        else:
            smiles_array = temp_df.loc[:, "smiles.std"].values
            rts = temp_df.loc[:, ["rt"]].values
            try:
                scaler, train_loader, val_loader, test_loader, test_indices = get_dataloaders(feature_array=smiles_array,
                                                                                          target_array=rts,
                                                                                          type="smiles",
                                                                                          split_by_scaffold=True)
                mpnn, trainer = configure_and_train_mpnn(scaler, train_loader, val_loader, param_dict, path2res, save_model=False)
                test_pred = trainer.predict(mpnn, test_loader)
                test_pred = np.concatenate(test_pred, axis=0)
                # GETTING RESULTS
                print(f"Getting results for the repo {cc_id}")
                temp_test_df = temp_df.iloc[test_indices[0]]
                temp_res_table = get_res_table(temp_test_df, test_pred, path2res, save_results=False,
                                               using_moldescs=using_moldescs)
                results_array.append(temp_res_table)
                del mpnn, trainer
            except:
                print (f"{cc_id} does not contain > 3 Bermis-Murcko scaffolds")
                cant_train_array.append(cc_id)
                continue

    cant_train_df = pd.DataFrame({"Contain not enough ms scaffolds":cant_train_array})
    path2cant_train = os.path.join(path2res, "cant_train.tsv")
    print (f"Writting the final result files in {path2res}...")
    cant_train_df.to_csv(path2cant_train, sep="\t", index=False)
    res_table = pd.concat (results_array, ignore_index=True)
    res_table.to_csv (path2res+ "Results.tsv", sep = "\t", index = False)
    mae, rmse, mre,rel_max_error, rel_mean_error = metrics_from_dataframe(res_table)
    write_parameters_file(param_dict, path2res)
    write_metrics_per_cc(res_table, path2res)
    write_metric_txt(mae, rmse, mre,rel_max_error, rel_mean_error, path2res)

    print ("The resuls written successfully! Exiting the program...")
    sys.exit(0)


