"""
Name: hpopting_functions.py
Author: Yixi Zhang
Version: 1.0.
Description: Contains functions for performing hyperparameters optimization for both SMRT and RepoRT.
"""

# IMPORT MODULES
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning import pytorch as pl
from chemprop import nn, models

# DEFINE THE SMRT objective function
def smrt_objective(trial,func,train_loader, val_loader, scaler):
    """
    This functino builds the backbone fot Optuna Hyperparameter optimization.
    """
    # Model
    config_dict = func(trial)
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

def report_objective(trial,func,train_loader, val_loader, scaler, cc_shape):
    """
    This functino builds the backbone fot Optuna Hyperparameter optimization.
    """
    # Model
    config_dict = func(trial)
    mp = nn.BondMessagePassing(d_h=config_dict["mp_hidden_dim"],
                               depth=config_dict["mp_depth"])
    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(output_transform=output_transform,
                           hidden_dim=config_dict["ffn_hidden_dim"],
                           input_dim=config_dict["mp_hidden_dim"] + cc_shape,
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

def write_hpop_params (search_space, path2res):
    """
    Write a txt file in the directory for saving results with the searching space defined in the file.
    """
    filename = path2res + "hpopt_params.txt"
    with open(filename, "w") as f:
        f.write ("The search space in this optimization run is:\n")
        for key, value in search_space.items():
            f.write (f"{key}: {value}\n")
def get_results_table_from_study (study, path2res):
    """
    Export an .tsv file with all the results got from a hyperparameter optimization run to the result saving directory.
    """
    filename = path2res + "Results_table.tsv"
    results = study.trials_dataframe()
    clean_results = results.drop (columns = ["number","datetime_start", "datetime_complete", "state"]).rename (columns = {"value":"val_loss"}) #Drop not interesting columns.
    sorted_results = clean_results.sort_values(by = ["val_loss"]).reset_index (drop = True) #Sort the DataFrame by the val_loss
    sorted_results.to_csv (filename, sep='\t', index = False)
    print (f"Successfully written the Result table and saved as {filename}")
