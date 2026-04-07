"""
Name: train_basic_model.py
Author: Yixi Zhang
Date: March 2026
Version: 1.0
This file contains the basic functions to train a chemprop model with no mol_descs.
"""

# 0. Import modules

import pandas as pd
import numpy as np
from chemprop import data, nn, models, featurizers
from rdkit.Chem.inchi import MolFromInchi
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import torch


# BUILD AND CONFIGURE chemprop.MPNN.
def get_dataloaders (feature_array, target_array, dataset = "SMRT"):
    """
    Input: a pandas dataframe containing the information for training.
    Outputs: Scaler used for scaling the targets. Train, val and test dataloaders built from the the dataframe. An array contaning the test indices.
    """
    # Convert the raw data to mol objects for getting the graph representation
    if dataset == "SMRT": #Mol objects from Inchi
        mols = [MolFromInchi(inchi, sanitize=False) for inchi in feature_array]
        all_data = [data.MoleculeDatapoint(mol, rt) for mol, rt in zip(mols, target_array)]  # DataPoints
    elif dataset == "RepoRT": #Build the mol object from SMILES
        all_data = [data.MoleculeDatapoint.from_smi(smi, rt) for smi, rt in zip(feature_array, target_array)]
        mols = [ d.mol for d in all_data ]
    else:
        print (f"Check the dataset given: {dataset}")
        return None

    #Get datapoints

    # Train_test_val_split. Splitting regarding the structure.
    train_indices, val_indices, test_indices = data.make_split_indices(
        mols,
        "random",
        (0.8, 0.1, 0.1),
        seed=42,
    )

    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )

    # Get DataSets
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    train_dset = data.MoleculeDataset(train_data[0], featurizer)
    scaler = train_dset.normalize_targets()

    val_dset = data.MoleculeDataset(val_data[0], featurizer)
    val_dset.normalize_targets(scaler)  # Scaling the targets.

    test_dset = data.MoleculeDataset(test_data[0], featurizer)

    # Dataloaders

    train_loader = data.build_dataloader(train_dset, num_workers=5, seed=42, persistent_workers=True)
    val_loader = data.build_dataloader(val_dset, num_workers=5, shuffle=False, persistent_workers=True)
    test_loader = data.build_dataloader(test_dset, num_workers=5, shuffle=False, persistent_workers=True)

    return scaler,train_loader, val_loader, test_loader, test_indices


def configure_and_train_mpnn (scaler, train_loader, val_loader, param_dict, results_path, save_model = True):
    """
    Used for configure and train a chemprop model using Trainer from lightning.pytorch
    """
    mp = nn.BondMessagePassing(
        d_h=param_dict["mp_hidden_dim"],
        depth=param_dict["mp_depth"],
    )
    agg = nn.MeanAggregation ()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN (output_transform = output_transform,
                            input_dim = param_dict["mp_hidden_dim"],
                            hidden_dim = param_dict ["ffn_hidden_dim"],
                            n_layers=param_dict ["ffn_layers"],
                            dropout=param_dict ["dropout_rate"],
                            criterion=nn.MSE (),
                            )
    mpnn = models.MPNN (mp, agg, ffn, param_dict["batch_norm"], param_dict ["metric_list"],
                        init_lr = param_dict["init_lr"],
                        max_lr = param_dict["max_lr"],
                        warmup_epochs=param_dict["warm_up_epochs"],
                        final_lr=param_dict["final_lr"],
                        )

    # 6. Training configuration

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
        accelerator=param_dict ["accelerator"],
        devices=1,
        max_epochs=param_dict ["max_epochs"],
        callbacks=[es_cb, checkpoint_cb],
    )

    # 7. Training

    trainer.fit (mpnn, train_loader, val_loader)
    if checkpoint_cb.best_model_path:
        mpnn = models.MPNN.load_from_checkpoint(checkpoint_cb.best_model_path)

    # 8. Saving the trained weights and biases only. The architecture will not be saved.
    if save_model:
        save_model_path = results_path + "model.pt"
        torch.save (mpnn.state_dict(), save_model_path)
    return mpnn, trainer

#model.load_state_dict(torch.load(PATH, weights_only=True)) #For loading. The exact same model should be first defined.

# RESULT FILES AND METRICS

def get_res_table_SMRT (df, pred_array, test_indices):
    """
    Input: An array containing InChi, another containing target and last one containing the prediction (Test set)
    Output: A pandas dataframe with the prediction table. The exportation should be done outside the function.
    """
    inchi_array = df.loc [:, "inchi"].values
    rt_array = df.loc [:, "rt"].values
    inchis = []
    real_rt = []
    for index in test_indices[0]:
        inchi = inchi_array[index]
        inchis.append(inchi)
        target = rt_array[index]
        real_rt.append(target)
    pred_list = []
    for res in pred_array:
        pred_list.append(res[0])
    real_rt = np.array(real_rt)
    pred_list = np.array(pred_list)
    res_table = pd.DataFrame ({ "InChi":inchis,
                                "real_rt": real_rt,
                                "pred_rt": pred_list,
                                "diff": np.abs (real_rt - pred_list),
                                })
    return res_table

def get_res_table_RepoRT (df, test_indices, pred_array):
    """
    Inputs: A pandas dataframe containing the data used for training. An array (2D) containing the test indices. An array containing the results from prediction.
    A filename (Absolute path + filename, e.g.). to save the .tsv file.
    Output: A pandas dataframe containing the results and the differences (Not sorted) and a saved .tsv file in the indicated dir path.
    """
    id_array = df.loc [:,"molecule_id"].values
    smiles = df.loc [:,"smiles.std"].values
    real_rts = df.loc [:,"rt_s"].values
    max_rts = df.loc [:,"max_rt"].values
    mean_rts = df.loc [:,"mean_rt"].values
    test_ids = []
    test_smiles = []
    test_rts = []
    pred_res = []
    max_array = []
    mean_array = []
    for index in test_indices [0]:
        id = id_array[index]
        smile = smiles[index]
        real_rt = real_rts[index]
        max_rt = max_rts[index]
        mean_rt = mean_rts[index]
        test_ids.append(id)
        test_smiles.append(smile)
        test_rts.append (real_rt)
        max_array.append (max_rt)
        mean_array.append (mean_rt)
    for res in pred_array:
        pred_res.append(round(res[0],2))
    test_rts = np.array(test_rts)
    pred_res = np.array(pred_res)
    max_rt = np.array(max_array)
    mean_rt = np.array(mean_array)
    result_table = pd.DataFrame ({ "id": test_ids,
                                    "smile": test_smiles,
                                    "real_rt": test_rts,
                                    "pred_rt": pred_res,
                                    "max_rt": max_array,
                                    "mean_rt": mean_array,
                                    "diff": np.abs (test_rts - pred_res),
                                    "rel_error_max_rt": np.abs (test_rts - pred_res)*100/max_array,
                                    "rel_error_mean_rt": np.abs (test_rts - pred_res)*100/mean_array
    })
    return result_table

def metrics_from_dataframe_RepoRT (df):
    """
    Input: DataFrame with "diff" column..
    Output: MAE and RMSE calculated from those values.
    """
    mae = np.mean (df["diff"])
    rmse = np.sqrt(np.mean (df["diff"] ** 2))
    mean_rel_error_max_rt = np.mean (df["rel_error_max_rt"])
    mean_rel_error_mean_rt = np.mean (df["rel_error_mean_rt"])
    return mae, rmse, mean_rel_error_max_rt, mean_rel_error_mean_rt
def metrics_from_dataframe (df):
    """
    Input: DataFrame with "diff" column..
    Output: MAE and RMSE calculated from those values.
    """
    mae = np.mean (df["diff"])
    rmse = np.sqrt(np.mean (df["diff"] ** 2))
    return mae, rmse

def write_metric_txt (mae, rmse, results_path):
    filename = results_path + "metrics.txt"
    with open (filename, "w") as f:
        f.write (f'MAE: {mae:.4f} s\nRMSE: {rmse:.4f} s.\n')

def write_metric_tsv (id_array, mae_array, rmse_array,relmax_array, relmean_array, results_path):
    """
    Given an id_array, mae_array and rmse_array, exports a tsv containing those data.
    """
    filename = results_path + "metrics.tsv"
    metrics_df = pd.DataFrame({"dir_id":id_array, "mae": mae_array, "rmse": rmse_array, "relative_error_to_max_rt (%)":relmax_array, "relative_error_to_mean_rt (%)":relmean_array})
    metrics_df.to_csv (filename, sep = "\t", index = False)

def write_parameters_file (param_dict, results_path):
    filename = results_path + "parameters.txt"
    with open (filename, "w") as f:
        f.write (f'Parameters used for this model:\n')
        for key,value in param_dict.items():
            f.write (f"{key}: {value}\n")





