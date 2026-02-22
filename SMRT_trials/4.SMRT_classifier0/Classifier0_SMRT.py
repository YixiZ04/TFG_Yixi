"""
4. Classifier0_SMRT
Author: Yixi Zhang
A binary classifier is constructed here with all default parameters from chemprop.
Here, those molecules whose RT < 5min (300s) are considered as non-retained (class 0).
Since it is very obvius to build and takes short time to train, neither of checkpoint nor model file are saved here.
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

# 1. Load Data
csv_data_file = "/SMRT_trials/SMRT_Data/SMRT_dataset.csv"
RT_data = pd.read_csv(csv_data_file, sep = ";")
RT_data ["retention"]  = [ 0 if rt < 300 else 1 for rt in RT_data["rt"] ] #Class 0 if rt < 300s else 1.
# 2. Get Data

inchis = RT_data.loc[:,"inchi"].values
retentions = RT_data.loc[:,["retention"]].values #Only class valor here

# 3. Processing data for training
mols = [ MolFromInchi(inchi, sanitize=False) for inchi in inchis ]

all_data = [data.MoleculeDatapoint (mol, retention) for mol, retention in zip(mols, retentions)] #DataPoints

mols_cp = [ d.mol for d in all_data ] # Mol object
train_indices, val_indices, test_indices = data.make_split_indices(mols_cp, "random", (0.8, 0.1, 0.1))

train_data, val_data, test_data = data.split_data_by_indices(
    all_data, train_indices, val_indices, test_indices
)
featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

train_dset = data.MoleculeDataset (train_data [0], featurizer)
val_dset = data.MoleculeDataset(val_data [0], featurizer)
test_dset = data.MoleculeDataset(test_data [0], featurizer)

train_loader = data.build_dataloader(train_dset, num_workers=5, persistent_workers = True)
val_loader = data.build_dataloader(val_dset, num_workers=5, shuffle=False, persistent_workers = True)
test_loader = data.build_dataloader(test_dset, num_workers=5, shuffle=False, persistent_workers = True)

# 4. Model

mp = nn.BondMessagePassing ()
agg = nn.MeanAggregation ()
ffn = nn.BinaryClassificationFFN ()
Batch_norm = False
metrics = None
model = models.MPNN (mp, agg, ffn, Batch_norm, metrics)

trainer = pl.Trainer (
    accelerator = "gpu", #Set to "auto" if not using GPU.
    devices = 1,
    max_epochs = 100,
)

trainer.fit (model, train_loader, val_loader)

#5. Get the results (ROC value)
test_red = trainer.test (model, dataloaders=test_loader, weights_only=False)
path_to_file = "C:/Users/leonz/Github_reps/TFG_git/SMRT_trials/4.SMRT_classifier0/results.txt"
with open(path_to_file, "w") as f:
    f.write(f"The roc of the model is: {test_red[0]['test/roc']:.4f}.")
