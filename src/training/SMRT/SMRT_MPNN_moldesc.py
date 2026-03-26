"""
Name: SMRT_MPNN_moldesc.py
Author: Yixi Zhang
Date: March 2026
Version: 1.1.
Usage: Train MPNN with SMRT dataset with molecular descriptors (mono_iso_mass and xlogp). Change the param_dict and saving dir path to avoid result overwriting.
It builds 4 output files in the directory defined for output:
    1. metrics.txt : A txt file containing the value og 2 metrics used for evaluating the model: MAE (s) and RMSE (s)
    2. model.pt: A .pt for saving the pytorch model.
    3. parameters.txt: A .txt file containing all parameters used for the training.
    4. Result_table.tsv: A .tsv file containing the results: pred_rt, real_rt, difference...
Update: Now if the input file does not exist, it will be built.
"""

# IMPORT MODULES

import os
import sys
from pathlib import Path
from lightning import pytorch as pl
from src.training.functions.basic_model_functions import write_metric_txt, metrics_from_dataframe, write_parameters_file
from src.training.functions.moldesc_model_functions import *
from src.get_molecular_descriptors.get_molecular_descriptors import add_columns2df


# DEFINE THE PARAMETERS. HERE DEFINED ARE THE DEFAULT VALUES OF CHEMPROP

input_file = Path("./data/with_extra_mol_desc/SMRT_extra_mol_descs.tsv")        # Be sure its existence.
path2res = "./logs/SMRT/moldesc/SMRT_moldesc_results_0/"                        # Change the dirname for each trial
param_dict = {
    "mp_hidden_dim": 300,                                                       # Hidden dimension of the message passing (MP) part
    "mp_depth": 3,                                                              # Depth/Number of Layers of the MP
    "ffn_hidden_dim": 300,                                                      # Hidden layer for the feed-forward network (ffn). This is the regressor
    "ffn_layers": 1,                                                            # Number of layers for the ffn.
    "init_lr": 1e-4,                                                            # The initial learning rate (lr)
    "max_lr": 1e-3,                                                             # Max lr will be reached in after the warm_up epochs.
    "final_lr": 1e-4,                                                           # The lr set for the rest of epochs.
    "warm_up_epochs": 2,                                                        # Number of epochs to reach the max_lr
    "max_epochs": 1000,                                                         # Set a huge number as early stopping mechanism is implemented here
    "dropout_rate": 0,                                                          # Dropout rate. 0 is default.
    "batch_norm": True,                                                         # True if want to apply batch_norm
    "metric_list": [nn.MAE(), nn.RMSE()],
    "accelerator": "auto",                                                      # If GPU and CUDA available change to "gpu". Or can set "cpu" as well.
}

# RUNNIG THE SCRIPT

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    print(f"Check for the dataset given...")
    if not input_file.exists():
        add_columns2df(path2dataset= "./data/no_extra_mol_desc/SMRT_data.csv", dataset="SMRT")
    df = pd.read_csv(input_file, sep="\t")
    # df = df.sample (500)        #Run this if want a quick test for usage
    print(f"Making the result directory...")
    os.makedirs(path2res, exist_ok=True)

    print(f"Getting DataLoaders and train chemprop model...")
    target_scaler, moldesc_scaler, train_loader, val_loader, test_loader, test_indices = get_dataloaders_with_moldesc (df, dataset="SMRT")
    mpnn, trainer = configure_and_train_mpnn_moldesc (target_scaler, moldesc_scaler, train_loader, val_loader, param_dict, path2res, save_model=True)

    print("Getting the predictions...")
    test_pred = trainer.predict(mpnn, test_loader)
    test_pred = np.concatenate(test_pred, axis=0)

    print("Making the result files...")
    write_parameters_file(param_dict, path2res)
    res_table = get_res_table_moldesc(df, test_pred, test_indices)
    res_table.to_csv(path2res + "Result_table.tsv", sep='\t', index=False)
    mae, rmse = metrics_from_dataframe(res_table)
    write_metric_txt(mae, rmse, path2res)
    print("Successfully trained the model and got the results!")
    sys.exit(0)
