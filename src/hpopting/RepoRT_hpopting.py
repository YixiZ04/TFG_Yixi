"""
Name: RepoRT_hpopting.py
Author: Yixi Zhang
Date: March 2026
Version: 1.0.
Usage: This file is used for doing hyperparameter optimization with SMRT dataset using Optuna. The range for the hyperparameters to tune should be defined.
Two result files are gotten from this Script:
    1. A txt file containing the searching space defined.
    2. A tsv file containing the results of the hyperparameter optimization.
IMPORTANT: split_type and dataset_type should be correctly defined. This version adapted the versions 1.2. for all Scripts from src.training.RepoRT.
And the results saving directory denpends on the split type and dataset_type, but the root dir is ./logs/hpopting/RepoRT/...
"""

# IMPORT MODULES

import os
import sys
import optuna
from pathlib import Path
from lightning import pytorch as pl
from optuna.search_space import intersection_search_space
from src.training.functions.splitted_sets_functions import *
from src.hpopting.hpopting_functions import *
from src.training.RepoRT.random_split.perform_random_splitting import split_train_val_test
from src.training.RepoRT.scaffold_split.perform_scaffold_split import ms_split
from src.training.RepoRT.cc_split.perform_cc_splitting import cc_split
from src.training.RepoRT.cc_scaffold_split.perform_cc_scaffold_split import cc_ms_split

# DEFINE THE SEARCH SPACE FOR THE OPTIMIZATION
num_trails = 2                                                                              # This is the numbers of trials to run, set to 2 for demonstration purpose.
dataset_type = "no_SMRT"                                                                    # Or "no_SMRT", depends on which dataset you want to use.
split_type = "random_split"                                                                 # "cc_split", "scaffold_split", "cc_scaffold_split"
apply_low_grad_filter = False                                                               # Set to True if want to apply low gradient filter.
input_dir = os.path.join (".", "data", "RepoRT_RP", "processed_data/", dataset_type, split_type + "/")
path2res = os.path.join (".", "logs", "hpopting","RepoRT_RP", dataset_type, split_type, "dirname/" )  # This is the result path to save the results. Change it when run a hyperparameter optimization.

def build_config (trial):
    """
    Here all the parameters are set to default values for a demonstration purpose.
    Change the searching space every time running a optimization.
    Also, a fixed value could be set if that hyperparameters is not wanted to be tuned in the run.
    """
    config_dict = {
        "mp_hidden_dim": trial.suggest_int("mp_hidden_dim", 300, 300, log=True),                                # Hidden dimension of the message passing (MP) part
        "mp_depth": trial.suggest_int("mp_depth", 3, 3, log=True),                                              # Depth/Number of Layers of the MP
        "ffn_hidden_dim": trial.suggest_int("ffn_hidden_dim", 300, 300, log=True),                              # Hidden layer for the feed-forward network (ffn). This is the regressor
        "ffn_layers": trial.suggest_int("ffn_layers", 3, 3, log=True),                                          # Number of layers for the ffn.
        "init_lr": trial.suggest_float("init_lr", 1e-4, 1e-4, log=True),                                        # The initial learning rate (lr)
        "max_lr": trial.suggest_float("max_lr", 1e-3, 1e-3, log=True),                                          # Max lr will be reached in after the warm_up epochs.
        "final_lr": trial.suggest_float("final_lr", 1e-4, 1e-4, log=True),                                      # The lr set for the rest of epochs.
        "warm_up_epochs": trial.suggest_int("warm_up_epochs", 2, 2, log=True),                                  # Number of epochs to reach the max_lr
        "max_epochs": 1000,                                                                                     # Set a huge number as early stopping mechanism is implemented here
        "dropout_rate": trial.suggest_float("dropout_rate", 0, 0),  # Dropout rate. 0 is default.
        "batch_norm": True,                                                                                     # True if want to apply batch_norm
        # "metric_list": [nn.MAE(), nn.RMSE()],                                                                 # Metric. Not really needed for this task.
        "accelerator": "auto",                                                                                  # If GPU and CUDA available change to "gpu". Or can set "cpu" as well.
    }
    return config_dict


# Start optimization process.
if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    train_file = Path(input_dir + "train_data.tsv")
    val_file = Path (input_dir + "val_data.tsv")
    test_file = Path (input_dir + "test_data.tsv")
    if train_file.exists() and val_file.exists() and test_file.exists():
        print ("The input files are correct!")
    else:
        print ("Building the input files...")
        match dataset_type:
            case "no_SMRT":
                complete_processed_file = os.path.join (".", "data", "RepoRT_RP","processed_data", "no_SMRT", "complete_processed_data.tsv")
                drop_smrt = True
            case "with_SMRT":
                complete_processed_file = os.path.join (".", "data", "RepoRT_RP", "processed_data","with_SMRT", "complete_processed_data.tsv")
                drop_smrt = False
            case _:
                raise NameError (f"The dataset given: {dataset_type} is not correct! Please check again...")
        match split_type:
            case "random_split":
                split_train_val_test(input_path = complete_processed_file, output_dir=input_dir, drop_smrt=drop_smrt, apply_low_grad_filter=apply_low_grad_filter)
            case "cc_split":
                cc_split(input_path = complete_processed_file, output_dir=input_dir, drop_smrt=drop_smrt, apply_low_grad_filter=apply_low_grad_filter)
            case "scaffold_split":
                ms_split(input_path = complete_processed_file, output_dir=input_dir, drop_smrt=drop_smrt, apply_low_grad_filter=apply_low_grad_filter)
            case "cc_scaffold_split":
                cc_ms_split(ms_complete_file = os.path.join (".", "data", "RepoRT_RP","processed_data", dataset_type, "ms_split/", "ms_complete_data.tsv"),
                            save_dir = input_dir,
                            random_seed =51,
                            processed_file=complete_processed_file,
                            drop_smrt=drop_smrt,
                            apply_low_grad_filter=apply_low_grad_filter,
                            save_complete_ms_dir=os.path.join (".", "data", "RepoRT_RP", "processed_data", dataset_type, "ms_split/"))
            case _:
                raise NameError (f"The split type given: {split_type} is not correct! Please check again...")
    print ("Getting the input dataframes...")
    train_df = pd.read_csv(train_file, sep="\t")
    val_df = pd.read_csv(val_file, sep="\t")
    test_df = pd.read_csv(test_file, sep="\t")

    # Uncomment this following block for a quick usage test.
    # train_df = train_df.sample (400)
    # val_df = val_df.sample (50)
    # test_df = test_df.sample (50)

    print(f"Making the result directory...")
    os.makedirs(path2res, exist_ok=True)

    print("Scaling the the metadata and gradient data using train Scaler...")
    train_df, train_input_scaler = get_scaled_input_train_data(train_df)
    val_df = get_scaled_datasets(val_df, train_input_scaler)
    test_df = get_scaled_datasets(test_df, train_input_scaler)

    print ("Getting the DataLoaders...")
    train_loader, scaler, cc_shape = get_train_dataloader(train_df)
    val_loader = get_val_loader(val_df, scaler, using_moldescs=False)
    test_loader = get_test_loader(test_df,using_moldescs=False)

    print ("Running hyperparameter optimization...")
    study = optuna.create_study (direction = "minimize")
    study.optimize (lambda trial:report_objective (trial, build_config, train_loader, val_loader, scaler, cc_shape=cc_shape),
                    n_trials = num_trails)
    print("Writing the configurartion txt file...")
    search_space = intersection_search_space(study.trials)
    write_hpop_params (search_space, path2res)
    print ("Getting the results...")
    get_results_table_from_study(study, path2res)

    sys.exit(0)
