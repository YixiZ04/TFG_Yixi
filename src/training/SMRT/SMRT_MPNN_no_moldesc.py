"""
Name: SMRT_MPNN_regression.py
Author: Yixi Zhang
Date: March 2026
Version: 1.0
Usage: Run this file to train MPNN for SMRT dataset without molecular descriptors. All the parameters displayed in the "param_dict" are the default parameters
from chemprop. Change values and the path to saving directory to avoid overwriting the results of previous runs if any.
The model has Early Stopping mechanism implemented.
"""

# IMPORT THE BASIC FUNCTIONS
from src.training.functions.basic_model_functions import *
import sys
import os
from lightning import pytorch as pl

# DEFINE THE PARAMETERS. HERE DEFINED ARE THE DEFAULT VALUES OF CHEMPROP
csv_data_file = "./data/no_extra_mol_desc/SMRT_data.csv"        #If not existing, it should be downloaded on internet. DO NOT USE FILE WITH MOL DESC
path2res = "./logs/SMRT/no_moldesc/SMRT_results_0/"                        # Change the dirname for each trial
param_dict = {
    "mp_hidden_dim": 300,                             # Hidden dimension of the message passing (MP) part
    "mp_depth": 3,                                    # Depth/Number of Layers of the MP
    "ffn_hidden_dim": 300,                            # Hidden layer for the feed-forward network (ffn). This is the regressor
    "ffn_layers": 1,                                  # Number of layers for the ffn.
    "init_lr": 1e-4,                                  # The initial learning rate (lr)
    "max_lr": 1e-3,                                   # Max lr will be reached in after the warm_up epochs.
    "final_lr": 1e-4,                                 # The lr set for the rest of epochs.
    "warm_up_epochs": 2,                              # Number of epochs to reach the max_lr
    "max_epochs": 1000,                               # Set a huge number as early stopping mechanism is implemented here
    "dropout_rate": 0,                                # Dropout rate. 0 is default.
    "batch_norm": True,                               # True if want to apply batch_norm
    "metric_list": [nn.MAE(), nn.RMSE()],
    "accelerator": "auto",                            # If GPU and CUDA available change to "gpu". Or can set "cpu" as well.
}


# Run the script
if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    print (f"Check for the dataset given...")
    try:
        df = pd.read_csv(csv_data_file, sep = ";")
        # df = df.sample (500)        #Run this if want a quick test for usage
        inchi_array = df.loc[:, "inchi"].values
        rts = df.loc[:, ["rt"]].values
    except FileNotFoundError:
        print (f"You should download this csv file from the Internet...")
        sys.exit (1)

    print (f"Making the result directory...")
    os.makedirs(path2res, exist_ok=True)

    print (f"Getting DataLoaders and train chemprop model...")
    scaler, train_loader, val_loader, test_loader, test_indices = get_dataloaders(feature_array=inchi_array,target_array=rts, dataset="SMRT")
    mpnn, trainer = configure_and_train_mpnn(scaler, train_loader, val_loader, param_dict, path2res, save_model=True)

    print ("Getting the predictions...")
    test_pred = trainer.predict(mpnn, test_loader)
    test_pred = np.concatenate(test_pred, axis=0)

    print ("Making the result files...")
    write_parameters_file(param_dict, path2res)
    res_table = get_res_table(df, test_pred, test_indices)
    res_table.to_csv (path2res + "Result_table.tsv", sep = '\t', index = False)
    mae, rmse = metrics_from_dataframe(res_table)
    write_metric_txt (mae, rmse, path2res)
    print ("Successfully trained the model and got the results!")
    sys.exit(0)
