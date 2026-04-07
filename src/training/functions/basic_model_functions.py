"""
Name: train_basic_model.py
Author: Yixi Zhang
Date: March 2026
Version: 1.1,
This file contains the basic functions to train a chemprop model with no mol_descs.
Update: Eliminated all functions related to RepoRT, as they are redundant comparing to splitted_sets_functions.py's functions
which are more specific and robust.

"""

# 0. Import modules

import pandas as pd
import numpy as np
import torch
from chemprop import data, nn, models, featurizers
from rdkit.Chem.inchi import MolFromInchi
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


# BUILD AND CONFIGURE chemprop.MPNN.
def get_dataloaders (feature_array, target_array):
    """
    Input: a pandas dataframe containing the information for training.
    Outputs: Scaler used for scaling the targets. Train, val and test dataloaders built from the the dataframe. An array contaning the test indices.
    """
    # Convert the raw data to mol objects for getting the graph representation
    mols = [MolFromInchi(inchi, sanitize=False) for inchi in feature_array]
    all_data = [data.MoleculeDatapoint(mol, rt) for mol, rt in zip(mols, target_array)]  # DataPoints

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

def get_res_table (df, pred_array, test_indices):
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

def write_parameters_file (param_dict, results_path):
    filename = results_path + "parameters.txt"
    with open (filename, "w") as f:
        f.write (f'Parameters used for this model:\n')
        for key,value in param_dict.items():
            f.write (f"{key}: {value}\n")





