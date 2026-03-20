"""
Name: splitted_sets_functions,py
Author: Yixi Zhang
Date: March 2026
Version: 1.1.
Description: contains the functions needed for training a MPNN (chemprop) with random split RepoRT data and write the result files.
To do so, run ./src/training/RepoRT/random_split/main.py
Update: A concept error was made in the previous version, as targets of each dataloader was scaled using a different Scaler. In this version, that has been fixed and a new function has been defined to
get val/test_loader using the train_scaler of the train dataset.
Updated metrics files, a new metric files will be created, it is a .tsv file containing the metrics (MAE, RMSE, %errors) for each repository contained in the test set.
IMPORTANT: The %errors in the "metrics.txt" IS NOT THE MEAN VALUE OF THIS NEW FILE, they are calculate as mean of the metrics calculated from molecule to molecule.
"""

#IMPORT MODULES
import numpy as np
import pandas as pd
from chemprop import data, nn, models, featurizers
from rdkit.Chem.inchi import MolFromInchi
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import torch

#DEFINE FUNCTIONS

def get_train_dataloader (train_df):
    """
    Input: The df containing the training data.
    Output: The train loader, the train_scaler and the cc_shape, which is the size of the metadata and the gradient data, used as input for training MPNN.
    Note that the shuffle Boolean has been set to True.
    """
    inchis = train_df.loc [:, "inchi.std"].values
    rts = train_df.loc [:, ["rt_s"]].values
    cc_columns = train_df.loc[:,"column.usp.code_L1":].columns
    temp_column_list = []
    for column in cc_columns:
        temp_list = train_df.loc [:, [column]].values
        temp_column_list.append (temp_list)
    cc = np.concatenate(
        temp_column_list, axis = 1
    )
    # BUILD MOL OBJECTS
    mols = [ MolFromInchi(inchi, sanitize=False) for inchi in inchis]
    #BUILD DATAPOINTS
    all_data = [[data.MoleculeDatapoint(mol, rt, x_d = X_d) for mol, rt, X_d in zip (mols, rts, cc) ]]
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer ()
    train_dset = data.MoleculeDataset (all_data[0] , featurizer)
    train_scaler = train_dset.normalize_targets()  # For the targets
    train_loader = data.build_dataloader(train_dset, num_workers=5, shuffle=True)
    return train_loader, train_scaler, len (temp_column_list)  #The last result is the cc_shape

def get_test_val_loader (df, train_scaler):
    """
    Input: the test or the val df and the scaler used for scaling training df.
    Output: the test or the val DataLoader obtained using the train_scaler, which can be obtained using the previous function.
    Note that here, the shuffle Boolean has been set to False.
    """
    inchis = df.loc[:, "inchi.std"].values
    rts = df.loc[:, ["rt_s"]].values
    cc_columns = df.loc[:, "column.usp.code_L1":].columns
    temp_column_list = []
    for column in cc_columns:
        temp_list = df.loc[:, [column]].values
        temp_column_list.append(temp_list)
    cc = np.concatenate(
        temp_column_list, axis=1
    )
    # BUILD MOL OBJECTS
    mols = [MolFromInchi(inchi, sanitize=False) for inchi in inchis]
    # BUILD DATAPOINTS
    all_data = [[data.MoleculeDatapoint(mol, rt, x_d=X_d) for mol, rt, X_d in zip(mols, rts, cc)]]
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dset = data.MoleculeDataset(all_data[0], featurizer)
    dset.normalize_targets(train_scaler)            #Here this is normalized with the train_scaler.
    data_loader = data.build_dataloader(dset, num_workers=5, shuffle=False) #Dependiong on if the val or test given.
    return data_loader

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
    temp_df = test_df [["dir_id","molecule_id", "inchi.std", "rt_s", "max_rt", "mean_rt"]]
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

def write_metrics_per_cc (res_df, result_path):
    """
    Input: The Result dataframe. (Created from "get_res_table")
    Usage: Get a .tsv file containing the mean per repository data metrics got before (MAE, RMSE and %error).
    """
    result = {
        "cc":[],
        "MAE":[],
        "RMSE": [],
        "Mean_relative_error_max":[],
        "Mean_relative_error_mean":[]
    }
    index_array = np.unique (res_df ["dir_id"])
    for index in index_array:
        temp_df = res_df [res_df ["dir_id"] == index]
        result ["cc"].append (index)
        result ["MAE"].append (np.mean (temp_df["diff"]))
        result ["RMSE"].append (np.sqrt (np.mean(temp_df["diff"] ** 2)))
        result ["Mean_relative_error_max"].append (np.mean (temp_df["rel_error_max"]))
        result ["Mean_relative_error_mean"].append (np.mean (temp_df["rel_error_mean"]))
    result = pd.DataFrame(result)
    result.to_csv (result_path + "metrics_per_cc.tsv", sep="\t", index=False)
    return

def write_metric_txt (mae, rmse, mean_rel_error_max_rt, mean_rel_error_mean_rt,results_path):
    filename = results_path + "metrics.txt"
    with open (filename, "w") as f:
        f.write (f'MAE: {mae:.4f} s\nRMSE: {rmse:.4f} s.\nRelative error to max rt (%): {mean_rel_error_max_rt:.4f}\nRelative error to mean rt (%): {mean_rel_error_mean_rt:.4f}\n')

def write_parameters_file(param_dict, results_path):
    filename = results_path + "parameters.txt"
    with open(filename, "w") as f:
        f.write(f'Parameters used for this model:\n')
        for key, value in param_dict.items():
            f.write(f"{key}: {value}\n")

