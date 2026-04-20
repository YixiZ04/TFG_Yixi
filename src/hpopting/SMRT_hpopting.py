"""
Name: SMRT_hpopting.py
Author: Yixi Zhang
Date: March 2026
Version: 1.1.
Usage: This file is used for doing hyperparameter optimization with SMRT dataset using Optuna. The range for the hyperparameters to tune should be defined.
Two result files are gotten from this Script:
    1. A txt file containing the searching space defined.
    2. A tsv file containing the results of the hyperparameter optimization.
Update (1.1.): Reorganized the structure of the file and the directory. But the usage remains the exact same. Also, an option to do hyperparameters optimization using
only retained molecules is given.
NOTE: The results would be saved in ./logs/hpopting/SMRT/...
"""

# IMPORT MODULES

import os
import sys
import optuna
import pandas as pd
from pathlib import Path
from lightning import pytorch as pl
from optuna.search_space import intersection_search_space
from src.training.functions.basic_model_functions import get_dataloaders
from src.hpopting.hpopting_functions import *


# DEFINE THE SEARCH SPACE FOR THE OPTIMIZATION
num_trails = 2                                                                                  # This is the numbers of trials to run, set to 2 for demonstration purpose.
save_dir_name = "dirname/"                                                                           # This is the result path to save the results. Change it when run a hyperparameter optimization.
csv_data_file = os.path.join (".", "data", "no_extra_mol_desc", "SMRT_data.csv")                # Path to the SMRT datafile.
only_retained = True                                                                            # Set to False if want to use all data.
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


# Start optimization process
if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    print(f"Check for the dataset given...")
    if Path (csv_data_file).exists():
        print ("The input dataset is correct...")
    else:
        print(f"You should download this csv file from the Internet...")
    df = pd.read_csv(csv_data_file, sep=";")
    if only_retained:
        df = df[df["rt"] > 300]       #Run this line if only retained mole
        path2res = os.path.join (".", "logs", "hpopting","SMRT", "only_retained", save_dir_name)
    else:
        path2res = os.path.join (".", "logs","hpopting",  "SMRT", "all_molecules", save_dir_name)
    # df = df.sample(500)  # Run this if want a quick test for usage
    inchi_array = df.loc[:, "inchi"].values
    rts = df.loc[:, ["rt"]].values
    print(f"Making the result directory...")
    os.makedirs(path2res, exist_ok=True)

    print(f"Getting DataLoaders and train chemprop model...")
    scaler, train_loader, val_loader, test_loader, test_indices = get_dataloaders(feature_array=inchi_array,target_array=rts)
    print ("Running hyperparameter optimization...")
    study = optuna.create_study (direction = "minimize")
    study.optimize (lambda trial:smrt_objective (trial, build_config, train_loader, val_loader, scaler),
                    n_trials = num_trails)
    print("Writing the configurartion txt file...")
    search_space = intersection_search_space(study.trials)
    write_hpop_params (search_space, path2res)
    print ("Getting the results...")
    get_results_table_from_study(study, path2res)

    sys.exit(0)
