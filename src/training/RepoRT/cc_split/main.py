"""
Name: training/RepoRT/cc_split/main.py
Author: Yixi Zhang.
Date: March 2026
Version: 1.0.
Usage: Run the file to train and configure a MPNN using chemprop and processed data randomly split from RepoRT.
If the input datafiles do not exist, they will be built.
"""



import os
import sys
from pathlib import Path
from src.training.functions.splitted_sets_functions import *
from src.training.RepoRT.cc_split.perform_cc_splitting import cc_split
# DEFINE PARAMETERS

train_file = Path ("./data/processed_RepoRT/random_split_data/train_data.tsv")
test_file = Path ("./data/processed_RepoRT/random_split_data/test_data.tsv")
val_file = Path ("./data/processed_RepoRT/random_split_data/val_data.tsv")
path2res = "./logs/RepoRT/cc_split_res/Results_try_1/"
param_dict = {
    "mp_hidden_dim": 300,                             # Hidden dimension of the message passing (MP) part
    "mp_depth": 3,                                    # Depth/Number of Layers of the MP
    "ffn_hidden_dim": 300,                            # Hidden layer for the feed-forward network (ffn). This is the regressor
    "ffn_layers": 1,                                  # Number of layers for the ffn.
    "init_lr": 1e-4,                                  # The initial learning rate (lr)
    "max_lr": 1e-3,                                   # Max lr will be reached in after the warm_up epochs.
    "final_lr": 1e-4,                                 # The lr set for the rest of epochs.
    "warm_up_epochs": 2,                              # Number of epochs to reach the max_lr
    "max_epochs": 40,                                 # Set to a smaller number as the datasets here are much smaller.
    "dropout_rate": 0.1,                                # Dropout rate. 0 is default.
    "batch_norm": True,                               # True if want to apply batch_norm
    "metric_list": [nn.MAE(), nn.RMSE()],
    "accelerator": "auto",                            # If GPU and CUDA available change to "gpu". Or can set "cpu" as well.
}

if __name__ == "__main__":
    if train_file.exists() and val_file.exists() and test_file.exists():
        print ("The input files are correct!")
    else:
        print ("Getting the input datasets.")
        cc_split() # This automatically get the files to the input path.
    train_df = pd.read_csv("./data/processed_RepoRT/cc_split_data/train_data.tsv", sep='\t').sample (400)
    test_df = pd.read_csv("./data/processed_RepoRT/cc_split_data/test_data.tsv", sep='\t').sample (50)
    val_df = pd.read_csv("./data/processed_RepoRT/cc_split_data/val_data.tsv", sep='\t').sample (50)

    print ("Input data are successfully read. Making the output directory.")
    os.makedirs (path2res, exist_ok=True)

    print ("Getting the DatLoaders...")
    val_loader, scaler,cc_shape = get_dataloader(val_df, shuffle=False)
    test_loader, scaler, cc_shape = get_dataloader(test_df, shuffle=False)
    train_loader, scaler, cc_shape = get_dataloader(train_df, shuffle=True)

    print ("Building and training the model...")
    mpnn, trainer = complete_cc_configure_train_model(scaler, train_loader, val_loader, param_dict, cc_shape = cc_shape,results_path=path2res, save_model=True)

    print (f"Writing the results files in {path2res}...")

    test_pred = trainer.predict(mpnn, test_loader)
    test_pred = np.concatenate(test_pred, axis=0)
    res_table = get_res_table(test_df, test_pred, path2res)
    mae, rmse, rel_max_error, rel_mean_error = metrics_from_dataframe(res_table)
    write_parameters_file(param_dict, path2res)
    write_metric_txt(mae, rmse, rel_max_error, rel_mean_error, path2res)

    print ("The resuls written successfully! Exiting the program...")
    sys.exit(0)