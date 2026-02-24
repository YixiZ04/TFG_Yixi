"""
Here, hyperparameter optimization is run with early stopping mechanism implemented. Search space:
    * ffn_hidden_dim = [300, 2000] (log = True)
    * mp_hidden_dim = [300, 600] (log = True)
    * ffn_layers = [2,6]
    * dropout_rate = [0.1, 0.2]
The depth is not tuned, but set to 3 (Default value). Because seem not to affect so much.
Also final_lr = 1e-6
"""

# 0.Import modules
import pandas as pd
from chemprop import data, nn, models, featurizers
from rdkit.Chem.inchi import MolFromInchi
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import optuna


# 1. Load SMRT data

# csv_data_file = "C:/Users/leonz/PyCharmMiscProject/TFG/SMRT_trials/SMRT_Data/SMRT_dataset.csv" #Windows path
csv_data_file = "/home/user/YixiTFG/TFG_Yixi/SMRT_trials/SMRT_Data/SMRT_dataset.csv"  #Linux Path
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
train_loader = data.build_dataloader (train_dset, num_workers = 6, shuffle = True)
val_loader = data.build_dataloader (val_dset, num_workers = 6, shuffle = False)


def objective(trial):
    # Hyperparameters to tune
    mp_hidden_dim = trial.suggest_int("mp_hidden_dim", 300, 600, log=True)  # 300 600
    ffn_hidden_dim = trial.suggest_int("ffn_hidden_dim", 300, 2000, log=True)  #300 2000
    ffn_layers = trial.suggest_int("ffn_layers", 2, 6)  # 2 6
    dropout_rate = trial.suggest_float("dropout_rate", 0, 0.2, step =0.1)
    # final_lr 1e-6

    # Model
    mp = nn.BondMessagePassing(d_h=mp_hidden_dim)
    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(output_transform=output_transform,
                           hidden_dim=ffn_hidden_dim,
                           input_dim=mp_hidden_dim,
                           n_layers=ffn_layers,
                           criterion=nn.MSE(),
                           dropout=dropout_rate,
                           )
    batch_norm = True
    metric_list = [nn.MAE(), nn.RMSE()]
    model = models.MPNN(mp, agg, ffn, batch_norm, metric_list, final_lr=1e-6)

    # Trainer
    checkpointing = ModelCheckpoint(
        # To get  the best val_loss from each model
        monitor="val_loss",
        mode="min",
    )
    early_stopping = EarlyStopping(
        monitor = "val_loss",
        mode = "min",
        patience = 10,
    )
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs = 1000,
        callbacks=[checkpointing, early_stopping],
    )
    trainer.fit(model, train_loader, val_loader)
    score = checkpointing.best_model_score
    val_loss = float("inf") if score is None else score.item()
    return val_loss

study = optuna.create_study (direction = "minimize")
study.optimize (objective, n_trials = 100)

#Obtein the result.tsv file for this optimization process

results = study.trials_dataframe()
clean_results = results.drop (columns = ["number","datetime_start", "datetime_complete", "state"]).rename (columns = {"value":"val_loss"}) #Drop not interesting columns.
sorted_results = clean_results.sort_values(by = ["val_loss"]).reset_index (drop = True) #Sort the DataFrame by the val_loss
sorted_results ["val_loss"] = round (sorted_results ["val_loss"], 4)

path2file = "/home/user/YixiTFG/TFG_Yixi/SMRT_trials/3.0.SMRT_hpop_advanced/3.0.5.SMRT_hpop_advanced5/Results_hpop_advanced5.tsv"
sorted_results.to_csv (path2file, sep='\t')
