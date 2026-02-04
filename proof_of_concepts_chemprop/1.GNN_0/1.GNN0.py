"""
1. GNN0
Author: Yixi Zhang
Here a basic GNN is constructed using all default values of the chemprop module:
    *  mp_hidden_dim = 300
    *  depth (mp) = 3
    *  ffn_hidden_dim = 300
    *  ffn_hidden_layers = 3
The data used here are fetched from RepoRT, in concrete, the first 40 Repos' data are used to train the GNN.
References:
    https://chemprop.readthedocs.io/en/main/training.html
    https://github.com/chemprop/chemprop/blob/main/examples/training.ipynb
"""

# 0.Import Modules
import pandas as pd
import numpy as np
from chemprop import data, featurizers, models, nn
from chemprop.nn import MeanAggregation
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

# 1. Fetch data from RepoRT

seed_url = 'https://raw.githubusercontent.com/michaelwitting/RepoRT/refs/heads/master/processed_data/'
def fetch_retention_time (base_url, rt_num):
    """
    Usage: Fetch data from RepoRT from the seed url (Only canonical SMILEs).
    Input: Number of how many repos you want to fetch.
    Output: A pd.Dataframe with the data.
    """
    # Empty DataTable to store Data.
    datatable = pd.DataFrame ()
    for i in range (1, rt_num+1):
        dir_num= str (i)
        while len (dir_num) < 4:
            dir_num = "0" + dir_num
        filename = base_url + dir_num + '/' + dir_num + '_rtdata_canonical_success.tsv'
        # Try to fetch data from the URL.
        try:
            temp_table = pd.read_csv(filename, sep='\t')
            datatable = pd.concat ([datatable,temp_table], ignore_index=True)
        # Except for an error (normally non-existing Repositorie)
        except:
            print ("Invalid url: " + filename)
    return datatable

# Get raw data. Only try to build the first ever GNN with few number of data.

raw_data_50 = fetch_retention_time (seed_url, 50) # Contains much more data than what we actually need.
smile_id_50 = raw_data_50.loc [:, "smiles.std"].values # SMILE (Our feature)
retention_time_50 = raw_data_50.loc [:, ["rt"]].values # RT (Targets)


# 2. Processing data for training
## 2.1.Datapoints

all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smile_id_50, retention_time_50)]

## 2.2. Convert to molecule form for better splitting

mols = [ d.mol for d in all_data ]

## 2.3. Splitting into train, valid, test sets

train_indices, val_indices, test_indices = data.make_split_indices(mols, "random", (0.8,0.1,0.1))

## 2.4. Get the data
train_data, val_data, test_data = data.split_data_by_indices(
    all_data, train_indices, val_indices, test_indices
)

## 2.5. Get MolecularDatasets
featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer ()

train_dset = data.MoleculeDataset (train_data [0], featurizer)
scaler = train_dset.normalize_targets()

val_dset = data.MoleculeDataset(val_data [0], featurizer)
val_dset.normalize_targets(scaler) #Scaling the targets.

test_dset = data.MoleculeDataset(test_data [0], featurizer)

## 2.6. DataLoaders.

train_loader = data.build_dataloader(train_dset, num_workers=5)
val_loader = data.build_dataloader(val_dset, num_workers=5, shuffle=False)
test_loader = data.build_dataloader(test_dset, num_workers=5, shuffle=False, persistent_workers = True)

# 3. Defining our model

## 3.1. Bond Message Passing

mp = nn.BondMessagePassing () # Only default values are used here (depth = 3, d_h = 300).

## 3.2. MeanAggregation

agg = MeanAggregation ()

## 3.3. Regression Neural Network

output_transform = nn.UnscaleTransform.from_standard_scaler(scaler) # Unscaling the results
ffn = nn.RegressionFFN (output_transform=output_transform) # Only default values are used

## 3.4. Batch Normalization (Boolean)

batch_norm = True

## 3.5. Metrics to report

metrics = [nn.metrics.MAE (), nn.metrics.RMSE (), nn.metrics.MSE ()]

## 3.6. Constructing the model

mpnn = models.MPNN (mp, agg, ffn, batch_norm, metrics,)

# 4. Training configurations

checkpointing = ModelCheckpoint (
    "C:/Users/leonz/PyCharmMiscProject/TFG/pruebas_concepto_chemprop/NeuralNetwork0_checkpoints",
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
    max_epochs = 50, #50 epochs are configurated here, only for proof of concept
    callbacks = [checkpointing],
)

# 5.Training precess

trainer.fit (mpnn, train_loader, val_loader)

# 6. Predicting

test_pred = trainer.predict (mpnn, test_loader)
test_pred = np.concatenate(test_pred, axis=0)

# 7. Result table.

def get_res_table (smile_array, target_array, pred_array, test_indices):
    """
    Input: An array containing the SMILEs, another containing target and last one containing the prediction.
    Output: A pandas dataframe with the prediction table.
    """
    smiles = []
    real_rt = []
    for index in test_indices[0]:
        smile = smile_array[index]
        smiles.append(smile)
        target = target_array[index][0]
        real_rt.append(target)
    pred_list = []
    for res in pred_array:
        pred_list.append(res[0])
    res_table = pd.DataFrame ({ "smiles":smiles,
                                "real_rt": real_rt,
                                "pred_rt": pred_list})
    return res_table

res_table = get_res_table (smile_id_50, retention_time_50, test_pred, test_indices)

# 8. Metrics
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
    MAE = (sum_num_MAE / m)*60
    RMSE = (np.sqrt(sum_num_RMSE ))*60
    return (MAE, RMSE)

filename = r'C:/Users/leonz/PyCharmMiscProject/TFG/res_GNN0.txt'
MAE, RMSE = MAE_RMSE_from_dataframe (res_table)
with open (filename, "w") as f:
    f.write (f'This file contains the result of GNN0 on test set.\n ')
    f.write (f'MAE: {MAE:.4f} s RMSE: {RMSE:.4f} s.\n')
    f.write (f'The result table is:\n {res_table.to_string()}.')

