"""
Name: training/RepoRT/random_split/main.py
Author: Yixi Zhang.
Date: March 2026
Version: 1.3.
Usage: Run the file to train and configure a MPNN using chemprop and processed data randomly split from RepoRT.
If the input datafiles do not exist, they will be built.
Update (1.1.): adapted to splitted_sets_functions.py version 1.1.
Update (1.2.): adapted to splitted_sets_functions.py version 1.2. (Input data scaling, aka metadatada and gradient data)
Update (1.3.): added an option to evaluate different datasets, for now, two types in total:
    1. All RepoRT data with SMRT removed. (no_SMRT)
    2. All RepoRT but downsampled those Repositories with >5000 molecuels (with_SMRT).
    If in the future, more datasets shuold be evaluated, this Script is easily extendable.
    (This could also be considered as adapting the version 1.2. of data_processing.py)
NOTE: Change dirname in line 28 to customize the saving directory name.
"""

#IMPORT MODULES
import os
import sys
from pathlib import Path
from src.training.functions.splitted_sets_functions import *
from src.training.RepoRT.random_split.perform_random_splitting import split_train_val_test

# DEFINE PARAMETERS

dataset_type = "no_SMRT" #Or with_SMRT, depends on the type of input dataset to use.
path2res = os.path.join(".", "logs","RepoRT", dataset_type, "random_split", "dirname/") #Change "dirname" for any name you want.
param_dict = {
    "mp_hidden_dim": 480,                             # Hidden dimension of the message passing (MP) part
    "mp_depth": 3,                                    # Depth/Number of Layers of the MP
    "ffn_hidden_dim": 1024,                            # Hidden layer for the feed-forward network (ffn). This is the regressor
    "ffn_layers": 5,                                  # Number of layers for the ffn.
    "init_lr": 1e-4,                                  # The initial learning rate (lr)
    "max_lr": 1e-3,                                   # Max lr will be reached in after the warm_up epochs.
    "final_lr": 1e-4,                                 # The lr set for the rest of epochs.
    "warm_up_epochs": 2,                              # Number of epochs to reach the max_lr
    "max_epochs": 1000,                                 # Set to a smaller number as the datasets here are much smaller.
    "dropout_rate": 0.1,                                # Dropout rate. 0 is default.
    "batch_norm": True,                               # True if want to apply batch_norm
    "metric_list": [nn.MAE(), nn.RMSE()],
    "accelerator": "auto",                            # If GPU and CUDA available change to "gpu". Or can set "cpu" as well.
}

if __name__ == "__main__":
    # Assertion for the data type. Match is used here for better generalization if in the future more dataset types will be evaluated.
    match dataset_type:
        case "no_SMRT":     #These following Booleans are used to get the processed dataset if has not been created yet.
            drop_smrt = True
            apply_upthreshold = False
            processed_filename = "no_SMRT_no_ds_data.tsv" #filename for the processed .tsv file.
        case "with_SMRT":
            drop_smrt = False
            apply_upthreshold = True
            processed_filename = "with_SMRT_ds_data.tsv"
        case _:
            raise NameError (f"Check the dataset_type: {dataset_type}.")

    # Define the processed file according to the dataset type given.
    input_file = os.path.join (".", "data", "processed_RepoRT", processed_filename)
    split_path = os.path.join (".", "data", "processed_RepoRT", dataset_type, "random_split_data/")
    train_file = Path (split_path + "train_data.tsv")
    test_file = Path (split_path + "test_data.tsv")
    val_file = Path (split_path + "val_data.tsv")

    # This conditions checks for the splitting files to exist. If NOT existing, they will be created.
    if train_file.exists() and val_file.exists() and test_file.exists():
        print ("The input files are correct!")
    else:
        print ("Getting the random_splitted files...")
        split_train_val_test(input_path= input_file, #Depending on the dataset that we want to evaluate, this function automatically creates the files needed.
                             output_dir=split_path,
                             drop_smrt=drop_smrt,
                             apply_upthreshold=apply_upthreshold)

    print ("Reading the input files...")
    train_df = pd.read_csv(train_file, sep='\t')
    test_df = pd.read_csv(test_file, sep='\t')
    val_df = pd.read_csv(val_file, sep='\t')

    print ("Input data are successfully read. Making the output directory...")
    os.makedirs (path2res, exist_ok=True)

    print("Scaling the the metadata and gradient data using train Scaler...")
    train_df, train_input_scaler = get_scaled_input_train_data(train_df)
    val_df = get_scaled_datasets(val_df, train_input_scaler)
    test_df = get_scaled_datasets(test_df, train_input_scaler)

    print ("Getting the DataLoaders...")
    train_loader, scaler, cc_shape = get_train_dataloader(train_df)
    val_loader = get_test_val_loader(val_df, scaler)
    test_loader = get_test_val_loader(test_df, scaler)

    print ("Building and training the model...")
    mpnn, trainer = complete_cc_configure_train_model(scaler, train_loader, val_loader, param_dict, cc_shape = cc_shape,results_path=path2res, save_model=True)

    print (f"Writing the results files in {path2res}...")

    test_pred = trainer.predict(mpnn, test_loader)
    test_pred = np.concatenate(test_pred, axis=0)
    res_table = get_res_table(test_df, test_pred, path2res)
    mae, rmse, rel_max_error, rel_mean_error = metrics_from_dataframe(res_table)
    write_parameters_file(param_dict, path2res)
    write_metrics_per_cc(res_table, path2res)
    write_metric_txt(mae, rmse, rel_max_error, rel_mean_error, path2res)

    print ("The resuls written successfully! Exiting the program...")
    sys.exit(0)
