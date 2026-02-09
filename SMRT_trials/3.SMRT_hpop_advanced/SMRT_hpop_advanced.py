"""
3. SMRT_hpop_advanced
Author: Yixi Zhang
Basic hyperparameters optimization is conducted using Optuna. These hyperparameters are optimized:
    1. mp_hidden_size: [300, 2400]
    2. ffn_hidden_size: [300, 2400]
    3. ffn_layers: [1,3]
    4. dropout_rate: [0.1, 0.5, step=0.1]
    5. init_lr: [1e-5, 1e-4, step=1e-5]
    6. max_lr: [1e-4, 1e-2, step=1e-3]
Warm_up_epochs used the default value: 2
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
import optuna

# 1. Load SMRT data

# csv_data_file = "C:/Users/leonz/PyCharmMiscProject/TFG/SMRT_trials/SMRT_Data/SMRT_dataset.csv" #Windows path
csv_data_file = "/home/user/YixiTFG/TFG_Yixi/SMRT_trials/SMRT_Data/SMRT_dataset.csv" #Linux Path
RT_data = pd.read_csv(csv_data_file, sep = ";") #Using all data

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
train_loader = data.build_dataloader (train_dset, num_workers = 5, shuffle = True)
val_loader = data.build_dataloader (val_dset, num_workers = 5, shuffle = False)

# 5. Define the objective function for hyperparameters optimization with Optuna

def objective (trial):

    # Hyperparameters to tune

    mp_hidden_dim = trial.suggest_int("mp_hidden_dim", 300, 2400, log=True)
    ffn_hidden_dim = trial.suggest_int ("ffn_hidden_dim", 300, 2400, log=True)
    ffn_layers = trial.suggest_int ("ffn_layers", 1, 3)
    dropout_rate = trial.suggest_float ("dropout_rate", 0.1, 0.5, step=0.1)
    init_lr = trial.suggest_float ("init_lr",1e-5, 1e-4, log=True, step=1e-5)
    max_lr = trial.suggest_float ("max_lr", 1e-3, 1e-2, log=True, step=1e-3)

    # Model
    mp = nn.BondMessagePassing(d_h=mp_hidden_dim)
    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(output_transform=output_transform,
                           hidden_dim=ffn_hidden_dim,
                           input_dim=mp_hidden_dim,
                           n_layers=ffn_layers,
                           criterion=nn.MSE (),
                           dropout=dropout_rate,
                           )
    batch_norm = True
    metric_list = [nn.MAE(), nn.RMSE()]
    model = models.MPNN(mp, agg, ffn, batch_norm, metric_list, init_lr=init_lr, max_lr=max_lr, final_lr=init_lr)

    # Trainer
    checkpointing = ModelCheckpoint (
        #To get  the best val_loss from each model
        monitor = "val_loss",
        mode = "min",
    )
    trainer = pl.Trainer(
        accelerator="auto",
        devices = 1,
        max_epochs = 20, #Set to 20 for optimization a grosso modo
        callbacks = [checkpointing],
    )
    trainer.fit (model, train_loader, val_loader)
    val_loss = trainer.checkpoint_callback.best_model_score.item ()
    return val_loss #(MSE)

# 6. Create Optuna study object

study = optuna.create_study (direction = "minimize")
study.optimize (objective, n_trials = 20)

#7. Get the results

results = study.trials_dataframe()
clean_results = results.drop (columns = ["number","datetime_start", "datetime_complete", "state"]).rename (columns = {"value":"val_loss"}) #Drop not interesting columns.
sorted_results = clean_results.sort_values(by = ["val_loss"]).reset_index (drop = True) #Sort the DataFrame by the val_loss
sorted_results ["val_loss"] = round (sorted_results ["val_loss"], 4)
#filename = "C:/Users/leonz/PyCharmMiscProject/TFG/SMRT_trials/Results_SMRT_hpop0.txt" #Windows Path name
filename = "/home/user/YixiTFG/TFG_Yixi/SMRT_trials/3.SMRT_hpop_advanced/Results_hpop_advanced.txt"
with open(filename, "w") as f:
    f.write (sorted_results.to_string())







