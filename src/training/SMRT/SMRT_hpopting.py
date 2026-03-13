"""
Name: SMRT_hpopting.py
Author: Yixi Zhang
Date: March 2026
Version: 1.0
Usage: This file is used for doing hyperparameter optimization with SMRT dataset using Optuna. The range for the hyperparameters to tune should be defined.
Two result files are gotten from this Script:
    1. A txt file containing the searching space defined.
    2. A tsv file containing the results of the hyperparameter optimization.
"""

# IMPORT MODULES

from lightning.pytorch.callbacks import ModelCheckpoint
from src.training.functions.basic_model_functions import *
import optuna
from optuna.search_space import intersection_search_space
import sys
import os

# DEFINE THE SEARCH SPACE FOR THE OPTIMIZATION
num_trails = 2                                                                                                  # This is the numbers of trials to run, set to 2 for demonstration purpose.
path2res = "./logs/SMRT/hpopting_results/hpopting_res_0/"                                                             # This is the result path to save the results. Change it when run a hyperparameter optimization.
csv_data_file = "./data/no_extra_mol_desc/SMRT_data.csv"
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


def objective(trial, train_loader, val_loader, scaler):
    """
    This functino builds the backbone fot Optuna Hyperparameter optimization.
    The search space is defined with the function "build_config_dict at the very beginning of the Script.
    """
    # Model
    config_dict = build_config(trial)
    mp = nn.BondMessagePassing(d_h=config_dict["mp_hidden_dim"],
                               depth=config_dict["mp_depth"])
    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(output_transform=output_transform,
                           hidden_dim=config_dict["ffn_hidden_dim"],
                           input_dim=config_dict["mp_hidden_dim"],
                           n_layers=config_dict["ffn_layers"],
                           criterion=nn.MSE(),                          #The val_loss set to MSE, change if want to use another loss function
                           dropout=config_dict ["dropout_rate"],
                           )
    model = models.MPNN(mp, agg, ffn, config_dict["batch_norm"],
                        init_lr=config_dict["init_lr"],
                        max_lr = config_dict["max_lr"],
                        warmup_epochs = config_dict["warm_up_epochs"],
                        final_lr=config_dict["final_lr"],)

    # Trainer
    checkpointing = ModelCheckpoint(
        # To get  the best val_loss from each model
        monitor="val_loss",
        mode="min",
    )
    early_stopping = EarlyStopping(
        monitor = "val_loss",
        mode = "min",
        patience = 10,
    )
    trainer = pl.Trainer(
        accelerator=config_dict["accelerator"],
        devices=1,
        max_epochs = config_dict["max_epochs"],
        callbacks=[checkpointing, early_stopping],
    )
    trainer.fit(model, train_loader, val_loader)
    score = checkpointing.best_model_score
    val_loss = float("inf") if score is None else score.item()
    return val_loss


# Start optimization process
if __name__ == "__main__":
    print(f"Check for the dataset given...")
    try:
        df = pd.read_csv(csv_data_file, sep=";")
        # df = df.sample (500)        #Run this if want a quick test for usage
        inchi_array = df.loc[:, "inchi"].values
        rts = df.loc[:, ["rt"]].values
    except FileNotFoundError:
        print(f"You should download this csv file from the Internet...")
        sys.exit(1)

    print(f"Making the result directory...")
    os.makedirs(path2res, exist_ok=True)

    print(f"Getting DataLoaders and train chemprop model...")
    scaler, train_loader, val_loader, test_loader, test_indices = get_dataloaders(feature_array=inchi_array,target_array=rts, dataset="SMRT")
    print ("Running hyperparameter optimization...")
    study = optuna.create_study (direction = "minimize")
    study.optimize (lambda trial:objective (trial, train_loader, val_loader, scaler),
                    n_trials = num_trails)
    print("Writing the configurartion txt file...")
    search_space = intersection_search_space(study.trials)
    write_parameters_file(search_space, path2res)
    print ("Getting the results...")
    get_results_table_from_study(study, path2res)

    sys.exit(0)