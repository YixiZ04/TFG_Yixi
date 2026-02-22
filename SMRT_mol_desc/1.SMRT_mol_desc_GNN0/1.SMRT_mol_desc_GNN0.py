"""
Author: Yixi Zhang
Here a GNN is constructed with default values from chemprop library.
Main reference for implementing >1 inputs to a chemprop model is this:
https://github.com/chemprop/chemprop/blob/main/examples/extra_features_descriptors.ipynb
"""

# 0. Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightning import pytorch as pl
from rdkit.Chem.inchi import MolFromInchi
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from chemprop import nn, models, featurizers, data

# 1. Loading Data

path2data = "/home/user/YixiTFG/TFG_Yixi/SMRT_mol_desc/SMRT_mol_desc_data/SMRT_mol_desc.csv"
datafile = pd.read_csv (path2data, sep=";")
rts = datafile.loc[:,["rt"]].values
inchis = datafile.loc [:,"inchi"].values
mol_weight = datafile.loc [:,["mol_weight"]].values
xlogp = datafile.loc [:,["xlogp"]].values #We want a matrix instead of a single array
datapoint_desc = np.concatenate ( #A matrix with shape 80000x2.
    [mol_weight, xlogp],axis=1
)

# 2. Get Mol object from inchi

mols = [ MolFromInchi(inchi, sanitize=False) for inchi in inchis ]

# 3.Molecular datapoint

datapoints = [ data.MoleculeDatapoint (mol,y,x_d=X_d) for mol,y,X_d in zip(mols,rts,datapoint_desc) ] #For each molecule, its weight and XlogP are added as extra features
updated_mol = [ d.mol for d in datapoints ] #Actually not necessary here, we can use the previous mol list
train_indices, val_indices, test_indices = data.make_split_indices(updated_mol, "random", (0.8, 0.1, 0.1))

train_data, val_data, test_data = data.split_data_by_indices(
    datapoints, train_indices, val_indices, test_indices
)

# DataSets

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer() #If atom or bond features were considered. We should add parameters here as well.

train_dset = data.MoleculeDataset (train_data [0], featurizer)
val_dset = data.MoleculeDataset(val_data [0], featurizer)
test_dset = data.MoleculeDataset(test_data [0], featurizer)

#Scalers. Train and Val dsets. We have to scale 2 different parameters: the targets and the features we introduced.

#Train_dset
targets_scaler = train_dset.normalize_targets() #For the targets
datapoint_desc_scaler = train_dset.normalize_inputs("X_d") #For the features

#Val_dset
val_dset.normalize_targets(targets_scaler)
val_dset.normalize_inputs("X_d", datapoint_desc_scaler)

# DataLoaders

train_loader = data.build_dataloader(train_dset, num_workers=5, shuffle=True)
val_loader = data.build_dataloader(val_dset, num_workers=5, shuffle=False)
test_loader = data.build_dataloader(test_dset, num_workers=5, shuffle=False)

# Build the model

mp = nn.BondMessagePassing ()
agg = nn.MeanAggregation ()
output_transform = nn.UnscaleTransform.from_standard_scaler(targets_scaler)
ffn_input_dim = mp.output_dim + datapoint_desc.shape [1] #Taking into account the feature dimension (2, in this case)
ffn = nn.RegressionFFN (input_dim=ffn_input_dim, output_transform = output_transform)
batch_norm = True
metrics = [ nn.MAE (), nn.RMSE ()]
X_d_transform = nn.ScaleTransform.from_standard_scaler(datapoint_desc_scaler) # Implement the scaler
model = models.MPNN (mp, agg, ffn, batch_norm, metrics, X_d_transform=X_d_transform) #The scaler for the extra features (X_d_transform)


# Set up the trainer
log_dir = "/home/user/YixiTFG/TFG_Yixi/SMRT_mol_desc/1.SMRT_mol_desc_GNN0/logs/"
logger = CSVLogger (
    save_dir=log_dir,
)

trainer = pl.Trainer(
    logger=logger,
    max_epochs=20,
    accelerator="auto", #It more or less converges.
    devices=1,
    enable_progress_bar=False,
)

trainer.fit (model, train_loader, val_loader)


# Prediction
test_pred = np.concatenate(trainer.predict (model, test_loader), axis=0)
# 4. Get results

## Results table

def get_res_table (inchi_array,mol_weight_array, xlogp_array, target_array, pred_array, test_indices):
    """
    Input: An array containing InChi, another containing target and last one containing the prediction (Test set)
    Output: A pandas dataframe with the prediction table.
    """
    inchis = []
    real_rt = []
    mol_weights = []
    xlogps = []
    pred_list = []
    for index in test_indices[0]:
        inchi = inchi_array[index]
        mol_weight = mol_weight_array [index][0]
        xlogp = xlogp_array [index][0]
        target = target_array[index][0]
        inchis.append(inchi)
        real_rt.append(target)
        mol_weights.append (mol_weight)
        xlogps.append (xlogp)
    for res in pred_array:
        pred_list.append(res[0])
    res_table = pd.DataFrame ({ "InChi":inchis,
                                "mol_weight":mol_weights,
                                "xlogp": xlogps,
                                "real_rt": real_rt,
                                "pred_rt": pred_list})
    return res_table

res_table = get_res_table (inchis, mol_weight, xlogp, rts, test_pred, test_indices)

#Export it in .tsv format
res_path = "/home/user/YixiTFG/TFG_Yixi/SMRT_mol_desc/1.SMRT_mol_desc_GNN0/Results/Results_GNN0.tsv"
res_table.to_csv (res_path, sep="\t", index=False)

#Get MAE and RMSE value from this table

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

MAE, RMSE = MAE_RMSE_from_dataframe (res_table)

metric_res_file = "/home/user/YixiTFG/TFG_Yixi/SMRT_mol_desc/1.SMRT_mol_desc_GNN0/Results/Results_metrics.txt"
with open (metric_res_file, "w") as f:
    f.write (f"MAE: {MAE:.4f} s\nRMSE: {RMSE:.4f} s")


## Val and train loss plot

def get_train_val_loss (path_to_log_file, sep=","):
    """
    Input: path to the log file (.csv format). The separator is the comma by defect.
    Output: 3 arrays: one for epochs, other for train loss and other for validation loss
    """
    loss_file = pd.read_csv (path_to_log_file, sep=sep)
    epochs = np.unique (loss_file ["epoch"])
    train_loss = loss_file ["train_loss_epoch"].dropna ().to_numpy()
    val_loss = loss_file ["val_loss"].dropna ().to_numpy ()
    return epochs, train_loss, val_loss

def save_val_train_loss_plot (epochs, train_loss, val_loss, path_to_save):
    plt.plot (epochs, train_loss, label="train_loss")
    plt.plot (epochs, val_loss, label="val_loss")
    plt.xlabel ("Epochs")
    plt.ylabel ("Scaled MSE loss")
    plt.grid (True)
    plt.legend ()
    plt.savefig (path_to_save+"train_val_loss.png", format="png")

logger_path = "/home/user/YixiTFG/TFG_Yixi/SMRT_mol_desc/1.SMRT_mol_desc_GNN0/logs/lightning_logs/version_0/metrics.csv"
res_path = "/home/user/YixiTFG/TFG_Yixi/SMRT_mol_desc/1.SMRT_mol_desc_GNN0/Results/"
epochs, train_loss, val_loss = get_train_val_loss(logger_path)
save_val_train_loss_plot (epochs, train_loss, val_loss, res_path)