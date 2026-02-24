"""
Main idea: use the data from every repository of RepoRT, use it to train a single model;and then store the results for every repo.
Basic MPNN is built here, this is, using all default parameters of chemprop.
    * mp_hidden_dim= 300
    * mp_depth = 3
    * ffn_n_layers = 1
    * ffn_hidden_dim = 300
    * final_lr = 1e-4
The train_val loss plot is not included for each repo data since EarlyStopping mechanism is implemented. Also, no checkpoints or model file are saved.
"""

# 0. Import modules

import pandas as pd
import numpy as np
from chemprop import data, nn, models, featurizers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning import pytorch as pl
import os

# 1. Define the functions we are gonna need
def get_index_array (num_repos):
    """
    Input: The number of repos you want to use for training
    Output: a numpy array containing the indexes string in the specific format ("0001", e.g.)
    """
    index_array = []
    for index in range(1, num_repos+1):
        index = str (index)
        while len (index) < 4:
            index = "0" + index
        index_array.append (str (index))
    return np.array(index_array)

def get_data_from_index (df, index):
    """
    Input: The raw dataframe containing all data and the index of the repo that you want to use for training.
    Output: A subset of the dataframe including all the molecules from the repo number.
    """
    temp_dataframe = df[df ["id"].str.split("_").str[0] == index]
    return temp_dataframe

def get_dataloader_from_df (df):
    """
    Input: a pandas dataframe containing the information for training.
    Outputs: Scaler used for scaling the targets. Train, val and test dataloaders built from the the dataframe. An array contaning the test indices.
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
        input_dim=mp_hidden_dim,
        hidden_dim = ffn_hidden_dim,
        n_layers = ffn_layers,
        output_transform = output_transform,
        criterion=nn.MSE(),
    )
    mpnn = models.MPNN (mp, agg, ffn, batch_norm, final_lr=final_lr)
    es_cb = EarlyStopping (patience=10, monitor="val_loss", mode="min")
    trainer = pl.Trainer (
        accelerator = "auto",
        devices = 1,
        max_epochs = 1000,
        callbacks = [es_cb],
    )
    print (f"Training a model with the default chemprop parameters...")
    trainer.fit (mpnn, train_loader, val_loader)

    #Get predictions
    res = np.concatenate (trainer.predict (mpnn, test_loader), axis=0)
    return res

def save_res_table (df, test_indices, pred_array, save_dir):
    """
    Inputs: A pandas dataframe containing the data used for training. An array (2D) containing the test indices. An array containing the results from prediction.
    A filename (Absolute path + filename, e.g.). to save the .tsv file.
    Output: A pandas dataframe containing the results and the differences (Not sorted) and a saved .tsv file in the indicated dir path.
    """
    id_array = df.loc [:,"id"].values
    smiles = df.loc [:,"smiles"].values
    real_rts = df.loc [:,"rt_s"].values
    test_ids = []
    test_smiles = []
    test_rts = []
    pred_res = []
    for index in test_indices [0]:
        id = id_array[index]
        smile = smiles[index]
        real_rt = real_rts[index]
        test_ids.append(id)
        test_smiles.append(smile)
        test_rts.append (real_rt)
    for res in pred_array:
        pred_res.append(round(res[0],2))
    result_table = pd.DataFrame ({
        "id": test_ids,
        "smile": test_smiles,
        "real_rt": test_rts,
        "pred_rt": pred_res,
        "diff": [round (np.abs (ps - ts),2) for ps,ts in zip(pred_res, test_rts)],
        "diff_sq": [round ((ps - ts)**2, 2) for ps,ts in zip(pred_res, test_rts)]
    })
    result_table.to_csv (save_dir, sep='\t', index=False)
    return result_table

def save_mae_rmse_form_datatable (res_table, save_file):
    """
    Input: DataFrame with target and predicted retention times.
    Output: MAE and RMSE calculated from those values.
    """
    diffs = res_table ["diff"]
    diff_sqs = res_table ["diff_sq"]
    n = len (diffs)
    MAE = np.sum (diffs) / n
    RMSE = np.sqrt (np.sum (diff_sqs) / n)
    return MAE, RMSE

def train_model_for_every_repo (df, save_dir, num_repos2train=450):
    """
    Inputs: A pandas dataframe containing all data. Seed name for the save dir ("/home/user/.../Results_). The num of the repos to use for training.
    Output: Saved results files in the repo for each index.
    Note: If a repo contains less than 20 molecules (empty repos included), will not be used for training as errors occur during training.
    """
    indexes = get_index_array(num_repos2train)
    for index in indexes:
        temp_df = get_data_from_index(df, index)
        if len(temp_df) < 20:
            print (f"The repo nº {index} is empty or not enough molecules are contained. It will be skipped")
            continue
        else:
            print (f"Starting training for repo {index}...")
            real_save_dir = save_dir +index+ "/"
            os.makedirs (real_save_dir, exist_ok=True)
            table_file = real_save_dir + "Results_" + index + ".tsv"
            metric_file = real_save_dir + "Metrics_" + index + ".txt"
            scaler, train_loader, val_loader, test_loader, test_indices = get_dataloader_from_df(temp_df)
            print (f"Successfully built the dataloaders for nº repo {index}...")
            temp_pred_res = configure_train_model(train_loader, val_loader, test_loader, scaler)
            res_table = save_res_table(temp_df, test_indices, temp_pred_res, table_file)
            MAE, RMSE = save_mae_rmse_form_datatable(res_table, metric_file)
            with open(metric_file, "w") as f:
                f.write(f"MAE: {MAE:.4f} s\nRMSE: {RMSE:.4f} s")
            print (f"The MPNN for the repo {index} has been trained successfully and the results are saved in {real_save_dir}.")
    print ("All models are successfully trained and the results are saved!")


# 2. Load Data

path2data = "/home/user/YixiTFG/TFG_Yixi/RepoRT_trials/RepoRT_data/RepoRT_data_clean.tsv"
df = pd.read_csv (path2data, sep='\t')
save_dir = "/home/user/YixiTFG/TFG_Yixi/RepoRT_trials/1.RepoRT_GNNs0/Results/Results_"
train_model_for_every_repo(df, save_dir, num_repos2train=450)


#temp_data = get_data_from_index(df, "0022")
#len (temp_data)
#scaler, train_loader, val_loader, test_loader, test_indices = get_dataloader_from_df(temp_data)
#temp_pred_res = configure_train_model(train_loader, val_loader, test_loader, scaler)
#res_table = save_res_table(temp_data, test_indices, temp_pred_res, save_dir)
