"""
Name: random_spliting_functions,py
Author: Yixi Zhang
Date: March 2026
Version: 1.0.
Description: contains the functions needed for training a MPNN (chemprop) with random split RepoRT data and write the result files.
To do so, run ./src/training/RepoRT/random_split/main.py

"""
#IMPORT MODULES
import pandas as pd
import numpy as np
from chemprop import data, nn, models, featurizers
from rdkit.Chem.inchi import MolFromInchi
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import torch


def get_dataloader (df, shuffle=True):
    """
    """
    inchis = df.loc [:, "inchi.std"].values
    rts = df.loc [:, ["rt_s"]].values
    cc_columns = df.loc[:,"column.usp.code_L1":].columns
    temp_column_list = []
    for column in cc_columns:
        temp_list = df.loc [:, [column]].values
        temp_column_list.append (temp_list)
    cc = np.concatenate(
        temp_column_list, axis = 1
    )
    # BUILD MOL OBJECTS
    mols = [ MolFromInchi(inchi, sanitize=False) for inchi in inchis]
    #BUILD DATAPOINTS
    all_data = [[data.MoleculeDatapoint(mol, rt, x_d = X_d) for mol, rt, X_d in zip (mols, rts, cc) ]]
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer ()
    dset = data.MoleculeDataset (all_data[0] , featurizer)
    targets_scaler = dset.normalize_targets()  # For the targets
    data_loader = data.build_dataloader(dset, num_workers=5, shuffle=shuffle)
    return data_loader, targets_scaler, len (temp_column_list)

def complete_cc_configure_train_model (scaler, train_loader, val_loader,param_dict, cc_shape, results_path, save_model=True):
    mp = nn.BondMessagePassing(d_h=param_dict["mp_hidden_dim"])
    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(output_transform=output_transform,
                           input_dim=param_dict["mp_hidden_dim"] + cc_shape,
                           hidden_dim=param_dict["ffn_hidden_dim"],
                           n_layers=param_dict["ffn_layers"],
                           dropout=param_dict["dropout_rate"],
                           criterion=nn.MSE(),
                           )
    mpnn = models.MPNN(mp, agg, ffn, param_dict["batch_norm"], param_dict["metric_list"],
                       init_lr=param_dict["init_lr"],
                       max_lr=param_dict["max_lr"],
                       warmup_epochs=param_dict["warm_up_epochs"],
                       final_lr=param_dict["final_lr"],
                       )

    # Training configuration

    es_cb = EarlyStopping(
        monitor="val_loss",  # In this case the val_loss is the MSE
        mode="min",
        patience=10,  # Patience set to 10
    )
    trainer = pl.Trainer(
        logger=False,
        enable_progress_bar=False,
        accelerator=param_dict["accelerator"],
        devices=1,
        max_epochs=param_dict["max_epochs"],
        callbacks=[es_cb],
    )

    # 7. Training

    trainer.fit(mpnn, train_loader, val_loader)

    # 8. Saving the trained weights and biases only. The architecture will not be saved.
    if save_model:
        save_model_path = results_path + "model.pt"
        torch.save(mpnn.state_dict(), save_model_path)
    return mpnn, trainer

def get_res_table (test_df, pred_array, save_dir):
    temp_df = test_df [["molecule_id", "inchi.std", "rt_s", "max_rt", "mean_rt"]]
    temp_df ["pred_rt"] = pred_array
    temp_df ["diff"] = np.abs (temp_df["pred_rt"] - temp_df["rt_s"])
    temp_df ["rel_error_max"] = temp_df ["diff"]*100 / temp_df["max_rt"]
    temp_df ["rel_error_mean"] = temp_df ["diff"]*100 / temp_df["mean_rt"]
    filename = save_dir + "Results.tsv"
    temp_df.to_csv(filename, sep="\t", index=False)
    return temp_df

def metrics_from_dataframe (df):
    """
    Input: DataFrame with "diff" column..
    Output: MAE and RMSE calculated from those values.
    """
    mae = np.mean (df["diff"])
    rmse = np.sqrt(np.mean (df["diff"] ** 2))
    mean_rel_error_max_rt = np.mean (df["rel_error_max"])
    mean_rel_error_mean_rt = np.mean (df["rel_error_mean"])
    return mae, rmse, mean_rel_error_max_rt, mean_rel_error_mean_rt

def write_metric_txt (mae, rmse, mean_rel_error_max_rt, mean_rel_error_mean_rt,results_path):
    filename = results_path + "metrics.txt"
    with open (filename, "w") as f:
        f.write (f'MAE: {mae:.4f} s\nRMSE: {rmse:.4f} s.\nRelative error to max rt (%){mean_rel_error_max_rt:.4f}\nRelative error to mean rt (%){mean_rel_error_mean_rt:.4f}\n')

def write_parameters_file(param_dict, results_path):
    filename = results_path + "parameters.txt"
    with open(filename, "w") as f:
        f.write(f'Parameters used for this model:\n')
        for key, value in param_dict.items():
            f.write(f"{key}: {value}\n")

