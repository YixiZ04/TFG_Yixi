"""
Main idea: use the data from every repository of RepoRT, use it to train a single model;and then store the results for every repo.
Basic MPNN is built here, this is, using all default parameters of chemprop.
    * mp_hidden_dim= 300
    * mp_depth = 3
    * ffn_n_layers = 1
    * ffn_hidden_dim = 300
    * final_lr = 1e-4
"""

# 0. Import modules

import pandas as pd
import numpy as np
from chemprop import data, nn, models, featurizers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning import pytorch as pl

# 1. Def training function
def get_data_from_index (index, df):
    temp_dataframe = df[df ["id"].str.split("_").str[0] == index]
    return temp_dataframe
def get_index_array (num_repos):
    index_array = []
    for index in range(num_repos):
        index = str (index)
        while len (index) < 4:
            index = "0" + index
    return np.array(index_array)

def get_data_from_each_repo (df, index):
    failed_model = 0
    temp_dataframe = get_data_from_index(index, df)
    if len (temp_dataframe) == 0:
        failed_model += 1
        print (f"The repo nº {index} is has no data.")
    else:
        print (f"Using data from repo {index} to train a chemprop model...")
        temp_dataframe = get_data_from_index(index, df)
    return temp_dataframe
def get_dataloader_from_df (df):
    """
    Train a chemprop model using data given from a pandas dataframe.
    As here we are only trying default values, no parameters are given as input to this function.
    """
    smiles = df.loc [:, "smiles"].values
    rts = df.loc [:, ["rt_s"]].values

    #Get datapoints
    all_data = [data.MoleculeDatapoint.from_smi(smi, rt) for smi, rt in zip(smiles, rts)]

    # Train_test_val_split

    mols = [ d.mol for d in all_data]
    train_indices, val_indices, test_indices = data.make_split_indices(mols, "random", (0.8, 0.1, 0.1))
    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )
    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )
    # Get molecule dataset
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    train_dset = data.MoleculeDataset (train_data [0], featurizer)
    scaler = train_dset.normalize_targets()

    val_dset = data.MoleculeDataset(val_data [0], featurizer)
    val_dset.normalize_targets(scaler) #Scaling the targets.

    test_dset = data.MoleculeDataset(test_data [0], featurizer)

    # Get DataLoaders.

    train_loader = data.build_dataloader(train_dset, num_workers=5)
    val_loader = data.build_dataloader(val_dset, num_workers=5, shuffle=False)
    test_loader = data.build_dataloader(test_dset, num_workers=5, shuffle=False)
    print ("The dataloaders successfully built...")
    return scaler,train_loader, val_loader, test_loader, test_indices

def configure_train_model (train_loader, val_loader, test_loader,scaler, mp_hidden_dim=300, ffn_layers=1, ffn_hidden_dim=300,final_lr=1e-4):
    """
    Input: configuration of a model. If none given, default values will be used.
    Output: Results for prediction.
    """
    mp = nn.BondMessagePassing (d_h=mp_hidden_dim)
    agg = nn.MeanAggregation ()
    batch_norm = True
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN (
        input_size=mp_hidden_dim,
        hidden_dim = ffn_hidden_dim,
        n_layers = ffn_layers,
        output_dim = output_transform,
        criterion=nn.MSE(),
    )
    mpnn = models.MPNN (mp, agg, ffn, batch_norm, final_lr=final_lr)
    es_cb = EarlyStopping (patience=10, monitor="val_loss", mode="min")
    trainer = pl.Trainer (
        accelerator = "auto",
        max_epochs = 1000,
        callbacks = [es_cb],
    )
    print (f"Training a model with the default chemprop parameters...")
    trainer.fit (mpnn, train_loader, val_loader)

    #Get predictions
    res = np.concatenate (trainer.predict (mpnn, test_loader), axis=0)
    return res

def save_res_table (df, test_indices, pred_array, save_dir):
    id_array = df ["id"]
    smiles = df ["smiles"]
    real_rts = df ["rt_s"]
    test_ids = []
    test_smiles = []
    test_rts = []
    for index in test_indices:
        id = id_array[index]
        smile = smiles[index]
        real_rt = real_rts[index]
        test_ids.append(id)
        test_smiles.append(smile)
        test_rts.append (real_rt)
    result_table = pd.DataFrame ({
        "id": test_ids,
        "smile": test_smiles,
        "real_rt": test_rts,
        "pred_rt": pred_array,
    })
    result_table.to_csv (save_dir, sep='\t', index=False)
    return result_table

def mae_rmse_form_datatable (res_table):
    """
        Input: DataFrame with target and predicted retention times.
        Output: MAE and RMSE calculated from those values.
        """
    sum_num_RMSE = 0
    sum_num_MAE = 0
    m = len(res_table["real_rt"])
    for i in range(m):
        diff = (res_table["real_rt"][i] - res_table["pred_rt"][i])
        diff_MAE = np.abs(diff)
        diff_RMSE = diff ** 2
        sum_num_RMSE += diff_RMSE
        sum_num_MAE += diff_MAE
    sum_num_RMSE = sum_num_RMSE / m
    MAE = (sum_num_MAE / m)
    RMSE = (np.sqrt(sum_num_RMSE))
    return (MAE, RMSE)

def get_results (df, )








# 2. Load Data

data_table = pd.read_csv ("C:/Users/leonz/Github_reps/TFG_git/RepoRT_trials/RepoRT_data/RepoRT_data_clean.tsv", sep="\t")