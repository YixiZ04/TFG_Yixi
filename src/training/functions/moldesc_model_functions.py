import pandas as pd
import numpy as np
from chemprop import data, nn, models, featurizers
from rdkit.Chem.inchi import MolFromInchi
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import torch

def get_dataloaders_with_moldesc (df, dataset = "SMRT"):
    """
    A DATAFRAME with molecular descriptors is required.
    And the dataset should be introduced for getting the correct molecule representation.
    Note: As in this version only extra two molecular descriptors are used and are the same for both datasets
    this function should be useful for both datasets.
    IMPORTANT: This function does not introduce the chromatograpy conditions, but only monisotopic_mass and xlogp.
    """
    # These are common for both datasets
    moniso_mws = df.loc[:, ["monoisotopic_mass"]].values
    xlogps = df.loc[:, ["xlogp"]].values
    mol_descs = np.concatenate([moniso_mws, xlogps], axis = 1)
    if dataset == "SMRT":  # Mol objects from Inchi
        inchi_array = df.loc[:, "inchi"].values
        rts = df.loc [:, ["rt"]].values
        mols = [MolFromInchi(inchi, sanitize=False) for inchi in inchi_array]
        all_data = [data.MoleculeDatapoint(mol, y, x_d=X_d) for mol, y, X_d in zip(mols, rts,mol_descs)]  # For each molecule, its weight and XlogP are added as extra features
    elif dataset == "RepoRT":  # Build the mol object from SMILES
        smiles = df.loc[:, "smiles.std"].values
        rts = df.loc [:, ["rt_s"]].values
        all_data = [data.MoleculeDatapoint.from_smi(smi, rt, x_d = X_d) for smi, rt, X_d in zip(smiles, rts, mol_descs)]
        mols = [d.mol for d in all_data]
    else:
        print(f"Check the dataset given: {dataset}")
        return None
    train_indices, val_indices, test_indices = data.make_split_indices(mols, "random", (0.8, 0.1, 0.1))

    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )

    # DataSets

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()  # If atom or bond features were considered. We should add parameters here as well.

    train_dset = data.MoleculeDataset(train_data[0], featurizer)
    val_dset = data.MoleculeDataset(val_data[0], featurizer)
    test_dset = data.MoleculeDataset(test_data[0], featurizer)

    # Scalers. Train and Val dsets. We have to scale 2 different parameters: the targets and the features we introduced.

    # Train_dset
    targets_scaler = train_dset.normalize_targets()  # For the targets
    mol_descs_scaler = train_dset.normalize_inputs("X_d")  # For the features

    # Val_dset
    val_dset.normalize_targets(targets_scaler)
    val_dset.normalize_inputs("X_d", mol_descs_scaler)

    # DataLoaders

    train_loader = data.build_dataloader(train_dset, num_workers=5, shuffle=True)
    val_loader = data.build_dataloader(val_dset, num_workers=5, shuffle=False)
    test_loader = data.build_dataloader(test_dset, num_workers=5, shuffle=False)

    return targets_scaler, mol_descs_scaler, train_loader, val_loader, test_loader, test_indices


def configure_and_train_mpnn_moldesc (target_scaler, mol_descs_scaler, train_loader, val_loader, param_dict, results_path, save_model = True):
    """
    Used for configure and train a chemprop model using Trainer from lightning.pytorch
    """
    # Build the model with the mol_desc_scaler

    mp = nn.BondMessagePassing (d_h = param_dict["mp_hidden_dim"])
    agg = nn.MeanAggregation ()
    output_transform = nn.UnscaleTransform.from_standard_scaler(target_scaler)
    X_d_transform = nn.ScaleTransform.from_standard_scaler(mol_descs_scaler)        #This is the different part.
    ffn = nn.RegressionFFN (output_transform = output_transform,
                            input_dim = param_dict["mp_hidden_dim"] + 2,            # Here is 2 because only 2 moldescs_were used.
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
                        X_d_transform=X_d_transform,
                        )

    # Training configuration

    es_cb = EarlyStopping(
        monitor="val_loss",  # In this case the val_loss is the MSE
        mode="min",
        patience=10,  # Patience set to 10
    )
    trainer = pl.Trainer(
        logger=False,
        enable_progress_bar=False,
        accelerator=param_dict ["accelerator"],
        devices=1,
        max_epochs=param_dict ["max_epochs"],
        callbacks=[es_cb],
    )

    # 7. Training

    trainer.fit (mpnn, train_loader, val_loader)

    # 8. Saving the trained weights and biases only. The architecture will not be saved.
    if save_model:
        save_model_path = results_path + "model.pt"
        torch.save (mpnn.state_dict(), save_model_path)
    return mpnn, trainer

# RESULTS

def get_res_table_moldesc_SMRT (df, pred_array, test_indices):
    """
    Input: An array containing InChi, another containing target and last one containing the prediction (Test set)
    Output: A pandas dataframe with the prediction table.
    """
    inchi_array = df.loc [:,"inchi"].values
    mol_weight_array = df.loc [:,"monoisotopic_mass"].values
    xlogp_array = df.loc [:,"xlogp"].values
    target_array = df.loc [:,"rt"].values
    inchis = []
    real_rt = []
    mol_weights = []
    xlogps = []
    pred_list = []
    for index in test_indices[0]:
        inchi = inchi_array[index]
        mol_weight = mol_weight_array [index]
        xlogp = xlogp_array [index]
        target = target_array[index]
        inchis.append(inchi)
        real_rt.append(target)
        mol_weights.append (mol_weight)
        xlogps.append (xlogp)
    for res in pred_array:
        pred_list.append(res[0])
    real_rt = np.array(real_rt)
    pred_list = np.array(pred_list)
    res_table = pd.DataFrame ({ "InChi":inchis,
                                "monoisotopic_mass": mol_weights,
                                "xlogp": xlogps,
                                "real_rt": real_rt,
                                "pred_rt": pred_list,
                                "diff": np.abs (pred_list - real_rt),})
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
    mol_weight_array = df.loc[:, "monoisotopic_mass"].values
    xlogp_array = df.loc[:, "xlogp"].values
    test_ids = []
    test_smiles = []
    test_rts = []
    pred_res = []
    mol_weights = []
    xlogps = []
    for index in test_indices [0]:
        id = id_array[index]
        smile = smiles[index]
        real_rt = real_rts[index]
        mol_weight = mol_weight_array[index]
        xlogp = xlogp_array[index]
        test_ids.append(id)
        test_smiles.append(smile)
        test_rts.append (real_rt)
        mol_weights.append(mol_weight)
        xlogps.append(xlogp)
    for res in pred_array:
        pred_res.append(round(res[0],2))
    test_rts = np.array(test_rts)
    pred_res = np.array(pred_res)
    result_table = pd.DataFrame ({ "id": test_ids,
                                    "smile": test_smiles,
                                    "monoisotopic_mass": mol_weights,
                                    "xlogp": xlogps,
                                    "real_rt": test_rts,
                                    "pred_rt": pred_res,
                                    "diff": np.abs (test_rts - pred_res),
    })
    return result_table




