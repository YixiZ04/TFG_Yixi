import pandas as pd
from chemprop import data, featurizers, models, nn
import torch
import numpy as np
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

seed_url = 'https://raw.githubusercontent.com/michaelwitting/RepoRT/refs/heads/master/processed_data/'
def fetch_retention_time (base_url, rt_num):
    datatable = pd.DataFrame ()
    for i in range (1, rt_num+1):
        dir_num= str (i)
        while len (dir_num) < 4:
            dir_num = "0" + dir_num
        filename = base_url + dir_num + '/' + dir_num + '_rtdata_canonical_success.tsv'
        try:
            temp_table = pd.read_csv(filename, sep='\t')
            datatable = pd.concat ([datatable,temp_table], ignore_index=True)
        except:
            print ("Invalid url: " + filename)
    return datatable
raw_data_100 = fetch_retention_time (seed_url, 100)
smile_id_100 = raw_data_100.loc [:, "smiles.std"].values
retention_time_100 = raw_data_100.loc [:, ["rt"]].values
## Processing data for training
### Datapoints
all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smile_id_100, retention_time_100)]

## Convert to molecule form for better splitting

mols = [ d.mol for d in all_data ]

## Splitting into train, valid, test sets

train_indices, val_indices, test_indices = data.make_split_indices(mols, "random", (0.8,0.1,0.1))

## Get the data
train_data, val_data, test_data = data.split_data_by_indices(
    all_data, train_indices, val_indices, test_indices
)

## Get MolecularDatasets
featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer ()

train_dset = data.MoleculeDataset (train_data [0], featurizer)
scaler = train_dset.normalize_targets()

val_dset = data.MoleculeDataset(val_data [0], featurizer)
val_dset.normalize_targets(scaler)

test_dset = data.MoleculeDataset(test_data [0], featurizer)

train_loader = data.build_dataloader(train_dset, num_workers=5)
val_loader = data.build_dataloader(val_dset, num_workers=5, shuffle=False)
test_loader = data.build_dataloader(test_dset, num_workers=5, shuffle=False, persistent_workers = True)

ckpt_path = "C:/Users/leonz/PycharmMiscProject/chemprop/checkpoints/best-epoch=0-val_loss=0.94.ckpt"
ckpt = ModelCheckpoint(ckpt_path)
model = MPNN.load_from_checkpoint(ckpt_path, weights_only= False)

with torch.inference_mode():
    trainer = pl.Trainer(
        logger=None,
        enable_progress_bar=True,
        accelerator="gpu",
        devices=1
    )
    test_preds = trainer.predict(model, test_loader)

test_preds = np.concatenate(test_preds, axis=0)
test_preds


