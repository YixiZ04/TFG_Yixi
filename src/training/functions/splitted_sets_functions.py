"""
Name: splitted_sets_functions,py
Author: Yixi Zhang
Date: March 2026
Version: 1.2.
Description: contains the functions needed for training a MPNN (chemprop) with random split RepoRT data and write the result files.
To do so, run ./src/training/RepoRT/random_split/main.py
Update (1.1.): A concept error was made in the previous version, as targets of each dataloader was scaled using a different Scaler. In this version, that has been fixed and a new function has been defined to
get val/test_loader using the train_scaler of the train dataset.
Updated metrics files, a new metric files will be created, it is a .tsv file containing the metrics (MAE, RMSE, %errors) for each repository contained in the test set.
IMPORTANT: The %errors in the "metrics.txt" IS NOT THE MEAN VALUE OF THIS NEW FILE, they are calculate as mean of the metrics calculated from molecule to molecule.
Update (1.2.): The functions to get the train scaler for the input (metadata and gradient data) and the scaled train set are defined here. The main difference in this new version
is that the input files is no longer scaled but are scaled just before training.
IMPORTANT: Here, the scaling data is not the whole dataset, but the metadata and gradient data for each repo are only considered 1 entree. In contrary, the input data
(metadata and gradient data) would be conditioned to the molecule number from each cc, but if this were not the better way to do the scaling, it will be changed in future versions.
"""

#IMPORT MODULES
import numpy as np
import pandas as pd

from chemprop import data, nn, models, featurizers
from rdkit.Chem.inchi import MolFromInchi
from sklearn.preprocessing import StandardScaler

import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

#DEFINE FUNCTIONS
def get_scaled_input_train_data (train_df):
    """
    Input: The train set used for training a model.
    Output: the scaled train set and the Scaler
    """

    temp_list = []
    train_input_scaler = StandardScaler ()
    index_array = np.unique (train_df["dir_id"])
    for index in index_array: #This loop is used to get the condition from each cc (chromatography condition)
        temp_df = train_df[train_df["dir_id"] == index]
        row = temp_df.iloc[:1]
        temp_list.append (row)
    temp_df = pd.concat(temp_list)
    input_data = temp_df.loc[:, "column.length":]
    train_input_scaler.fit (input_data) #This would be returned as output.
    position = train_df.columns.get_loc("column.length")
    columns_used = train_df.columns [position:]
    train_df [columns_used] = train_input_scaler.transform(train_df [columns_used])
    return train_df, train_input_scaler

def get_scaled_datasets (df, train_input_scaler):
    """
    Used for scaling the input df using the train Scaler.
    In this context, this is used to get test and val data using train_input_scaler.
    """
    position = df.columns.get_loc("column.length")
    columns_used = df.columns[position:]
    df [columns_used] = train_input_scaler.transform(df[columns_used])
    return df

def scaling_meta_grad_data (updated_df):
    """
    Requieres a df that only contains 1 metadata and gradient data per repo.
    Returns the same dataframe but scaled.
    """
    temp_df = updated_df.loc[:, "column.length":].astype(float)
    std_scaler = StandardScaler()
    scaled_array = std_scaler.fit_transform(temp_df)
    for column, array in zip (temp_df.columns, scaled_array.transpose()):
        temp_df[column] = array
    temp_df.insert(0, "dir_id", updated_df["dir_id"])
    return temp_df

def get_complete_scaled_df (df, updated_df):
    """
    Use the previous unscaled dataframe and the scaled metadata and gradient dataframe.
    And merges them together to get the complete scaled dataframe.
    """
    temp_df = df.loc[:, :"column.usp.code_nan"]
    final_df = pd.merge (temp_df, updated_df, on = "dir_id", how = "inner")
    return final_df


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
    train_loader = data.build_dataloader(train_dset, num_workers=5, shuffle=True, seed=42)
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
    mp = nn.BondMessagePassing(
        d_h=param_dict["mp_hidden_dim"],
        depth=param_dict["mp_depth"],
    )
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
    checkpoint_cb = ModelCheckpoint(
        dirpath=results_path,
        filename="best-{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    trainer = pl.Trainer(
        logger=False,
        enable_progress_bar=False,
        accelerator=param_dict["accelerator"],
        devices=1,
        max_epochs=param_dict["max_epochs"],
        callbacks=[es_cb, checkpoint_cb],
    )

    # 7. Training

    trainer.fit(mpnn, train_loader, val_loader)
    if checkpoint_cb.best_model_path:
        mpnn = models.MPNN.load_from_checkpoint(checkpoint_cb.best_model_path)

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
