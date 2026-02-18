"""
1. GNN0_SMRT
Author: Yixi Zhang
Here a basic GNN is constructed using all default values of the chemprop module:
    *  mp_hidden_dim = 300
    *  depth (mp) = 3
    *  ffn_hidden_dim = 300
    *  ffn_hidden_layers = 3
Hyperparameters optimization process with Optuna will be applied to this model and the results will be compared.
The data used here is from METLIN SMRT (Domingo-Almenara et al., 2019).
"""

# 0. Import modules

import pandas as pd
import numpy as np
from chemprop import data, nn, models, featurizers
from rdkit.Chem.inchi import MolFromInchi
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import torch

# 1. Load SMRT data

csv_data_file = "C:/Users/leonz/PyCharmMiscProject/TFG/SMRT_trials/SMRT_Data/SMRT_dataset.csv"
RT_data = pd.read_csv(csv_data_file, sep = ";")

# 2. Get Data

rts = RT_data.loc[:,["rt"]].values
inchis = RT_data.loc[:,"inchi"].values

# 3. Convert InChi to RDkit mol object.

mols = [ MolFromInchi(inchi, sanitize=False) for inchi in inchis ]

# 4. Preprocessing data for training

all_data = [data.MoleculeDatapoint (mol, rt) for mol, rt in zip(mols, rts)] #DataPoints

mols_cp = [ d.mol for d in all_data ] # Mol object

# Splitting

train_indices, val_indices, test_indices = data.make_split_indices(mols_cp, "random", (0.8, 0.1, 0.1))

train_data, val_data, test_data = data.split_data_by_indices(
    all_data, train_indices, val_indices, test_indices
)

# DataSets

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

train_dset = data.MoleculeDataset (train_data [0], featurizer)
scaler = train_dset.normalize_targets()

val_dset = data.MoleculeDataset(val_data [0], featurizer)
val_dset.normalize_targets(scaler) #Scaling the targets.

test_dset = data.MoleculeDataset(test_data [0], featurizer)

# DataLoaders

train_loader = data.build_dataloader(train_dset, num_workers=5, persistent_workers = True)
val_loader = data.build_dataloader(val_dset, num_workers=5, shuffle=False, persistent_workers = True)
test_loader = data.build_dataloader(test_dset, num_workers=5, shuffle=False, persistent_workers = True)

# 5. Configuration of MPNN

mp = nn.BondMessagePassing ()
agg = nn.MeanAggregation ()
output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
ffn = nn.RegressionFFN (output_transform = output_transform)
batch_norm = True
metrics = [ nn.MAE (), nn.RMSE ()]
mpnn = models.MPNN (mp, agg, ffn, batch_norm, metrics)

# 6. Training configuration

checkpointing = ModelCheckpoint (
    "C:/Users/leonz/PyCharmMiscProject/TFG/SMRT_trials/checkpoints",
    "best_{epoch}",
    "val_loss",
    mode = "min",
    save_last = True,
)

trainer = pl.Trainer (
    logger = False,
    enable_checkpointing = True,
    enable_progress_bar = True,
    accelerator = "gpu",
    devices = 1,
    max_epochs = 100,
    callbacks = [checkpointing],
)

# Training

trainer.fit (mpnn, train_loader, val_loader)

# Saving the trained weights and biases only. The architecture will not be saved.

saving_path = "C:/Users/leonz/PyCharmMiscProject/TFG/SMRT_trials/saved_model.pt"
torch.save (mpnn.state_dict(), saving_path)

#model.load_state_dict(torch.load(PATH, weights_only=True)) #For loading. The exact same model should be first defined.

# 7. Predicting

test_pred = trainer.predict (mpnn, test_loader)
test_pred = np.concatenate(test_pred, axis=0)

# 8. Result table.

def get_res_table (inchi_array, target_array, pred_array, test_indices):
    """
    Input: An array containing InChi, another containing target and last one containing the prediction (Test set)
    Output: A pandas dataframe with the prediction table.
    """
    inchis = []
    real_rt = []
    for index in test_indices[0]:
        inchi = inchi_array[index]
        inchis.append(inchi)
        target = target_array[index][0]
        real_rt.append(target)
    pred_list = []
    for res in pred_array:
        pred_list.append(res[0])
    res_table = pd.DataFrame ({ "InChi":inchis,
                                "real_rt": real_rt,
                                "pred_rt": pred_list})
    return res_table

res_table = get_res_table (inchis, rts, test_pred, test_indices)

# 9. Metrics

def MAE_RMSE_from_dataframe (dataframe):
    """
    Input: DataFrame with target and predicted retention times.
    Output: MAE and RMSE calculated from those values.
    """
    sum_num_RMSE = 0
    sum_num_MAE = 0
    m = len(dataframe ["real_rt"])
    for i in range (m):
        diff = (dataframe["real_rt"][i] - dataframe["pred_rt"][i])
        diff_MAE =  np.abs (diff)
        diff_RMSE = diff ** 2
        sum_num_RMSE += diff_RMSE
        sum_num_MAE += diff_MAE
    sum_num_RMSE = sum_num_RMSE / m
    MAE = (sum_num_MAE / m)
    RMSE = (np.sqrt(sum_num_RMSE ))
    return (MAE, RMSE)

# 10. Get the result file

filename = r'C:/Users/leonz/PyCharmMiscProject/TFG/SMRT_trials/results_GNN0.txt'
MAE, RMSE = MAE_RMSE_from_dataframe (res_table)
with open (filename, "w") as f:
    f.write (f'This file contains the result of GNN0 on test set.\n ')
    f.write (f'MAE: {MAE:.4f} s RMSE: {RMSE:.4f} s.\n')
    f.write (f'The result table is:\n {res_table.to_string()}.')
