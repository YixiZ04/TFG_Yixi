"""
Name: training/RepoRT/cc_scaffold_split/main.py
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
Update (1.4); added the option to train model with molecular descriptors.
"""

#IMPORT MODULES
import os
import sys
from pathlib import Path
from lightning import pytorch as pl
from src.training.functions.splitted_sets_functions import *
from src.training.RepoRT.cc_scaffold_split.perform_cc_scaffold_split import cc_ms_split

# DEFINE PARAMETERS
SOURCE_PATH = os.path.join(".", "data", "RepoRT_RP", "processed_data/")                        # This is the source directory that contains all processed files
dataset_type = "no_SMRT"                                                                  # Or with_SMRT, depends on the type of input dataset to use.
apply_grad_down_threshold = False                                                           # Set to True if want to use the filtered by grad_down_threshold
filtering = "filtered" if apply_grad_down_threshold else "no_filtered"
using_moldescs = False                                                                      # Set to True if want to use molecular descriptors for the model
moldesc_dir = "RepoRT_moldesc" if using_moldescs else "RepoRT_RP"                              # Changes the path where to save the results files
path2res = os.path.join(".", "logs", moldesc_dir, dataset_type, filtering, "cc_scaffold_split", "01_17_04_2026/") #Change "dirname" for any name you want.
path2moldesc = os.path.join (".", "data", "with_extra_mol_desc", "extra_mol_descs.tsv")


param_dict = {
    "mp_hidden_dim": 451,                             # Hidden dimension of the message passing (MP) part
    "mp_depth": 4,                                    # Depth/Number of Layers of the MP
    "ffn_hidden_dim": 1493,                            # Hidden layer for the feed-forward network (ffn). This is the regressor
    "ffn_layers": 4,                                  # Number of layers for the ffn.
    "init_lr": 1e-4,                                  # The initial learning rate (lr)
    "max_lr": 1e-3,                                   # Max lr will be reached in after the warm_up epochs.
    "final_lr": 1e-4,                                 # The lr set for the rest of epochs.
    "warm_up_epochs": 2,                              # Number of epochs to reach the max_lr
    "max_epochs": 1000,                               # Set to a smaller number as the datasets here are much smaller.
    "dropout_rate": 0.1,                              # Dropout rate. 0 is default.
    "batch_norm": True,                               # True if want to apply batch_norm
    "metric_list": [nn.MAE(), nn.RMSE()],
    "accelerator": "auto",                            # If GPU and CUDA available change to "gpu". Or can set "cpu" as well.
}


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    # Assertion for the data type. Match is used here for better generalization if in the future more dataset types will be evaluated.
    match dataset_type:
        case "no_SMRT":     #These following Booleans are used to get the processed dataset if has not been created yet.
            drop_smrt = True
            if apply_grad_down_threshold:
                path2input = os.path.join(SOURCE_PATH, "no_SMRT_down_grad_filter/")
            else:
                path2input = os.path.join(SOURCE_PATH, "no_SMRT/")
        case "with_SMRT":
            drop_smrt = False
            if apply_grad_down_threshold:
                path2input = os.path.join(SOURCE_PATH, "with_SMRT_down_grad_filter/")
            else:
                path2input = os.path.join(SOURCE_PATH, "with_SMRT/")
        case _:
            raise NameError (f"Check the dataset_type: {dataset_type}.")

    random_seed = 51
    # Define the processed file according to the dataset type given.
    input_file = os.path.join (path2input, "complete_processed_data.tsv")
    ms_complete_save_dir = os.path.join (path2input, "ms_split/")
    ms_complete_file = os.path.join (ms_complete_save_dir,"ms_complete_data.tsv")
    split_path = os.path.join (path2input, "cc_scaffold_split/")
    train_file = Path (split_path + "train_data.tsv")
    test_file = Path (split_path + "test_data.tsv")
    val_file = Path (split_path + "val_data.tsv")
    dir_report_file = Path (split_path + "dir_assignment_report.tsv")
    scaffold_report_file = Path (split_path + "scaffold_pruning_report.tsv")

    # This conditions checks for the splitting files to exist. If NOT existing, they will be created.
    if (
        train_file.exists()
        and val_file.exists()
        and test_file.exists()
        and dir_report_file.exists()
        and scaffold_report_file.exists()
    ):
        print ("The input files are correct!")
    else:
        print ("Getting the random_splitted files...")
        cc_ms_split (ms_complete_file= ms_complete_file, #Depending on the dataset that we want to evaluate, this function automatically creates the files needed.
                     save_dir=split_path,
                     random_seed=random_seed,
                     processed_file=input_file,
                     save_complete_ms_dir=ms_complete_save_dir,
                     drop_smrt=drop_smrt,
                     apply_low_grad_filter=apply_grad_down_threshold)

    print ("Reading the input files...")
    train_df = pd.read_csv(train_file, sep='\t')
    test_df = pd.read_csv(test_file, sep='\t')
    val_df = pd.read_csv(val_file, sep='\t')

    if (
        train_df.empty
        or val_df.empty
        or test_df.empty
        or not dir_report_file.exists()
        or not scaffold_report_file.exists()
    ):
        print ("Found empty split files. Rebuilding the constrained split...")
        cc_ms_split (ms_complete_file= ms_complete_file,
                     save_dir=split_path,
                     random_seed=random_seed,
                     processed_file=input_file,
                     save_complete_ms_dir=ms_complete_save_dir,
                     drop_smrt=drop_smrt,
                     apply_low_grad_filter=apply_grad_down_threshold)
        train_df = pd.read_csv(train_file, sep='\t')
        test_df = pd.read_csv(test_file, sep='\t')
        val_df = pd.read_csv(val_file, sep='\t')

    print ("Input data are successfully read. Making the output directory...")
    os.makedirs (path2res, exist_ok=True)

    if using_moldescs:
        print ("Scaling the molecular descriptors...")
        train_df = add_moldescs(train_df, path2moldesc)
        test_df = add_moldescs(test_df, path2moldesc)
        val_df = add_moldescs(val_df, path2moldesc)
        scaled_train_df, moldesc_scaler = get_scaled_moldescs_train(train_df)
        scaled_test_df = get_scaled_moldesc_testval(test_df, moldesc_scaler)
        scaled_val_df = get_scaled_moldesc_testval(val_df, moldesc_scaler)
        print("Scaling the the metadata and gradient data using train Scaler...")
        scaled_train_df, train_input_scaler = get_scaled_input_train_data(scaled_train_df)
        scaled_val_df = get_scaled_datasets(scaled_val_df, train_input_scaler)
        scaled_test_df = get_scaled_datasets(scaled_test_df, train_input_scaler)
    else:
        print("Scaling the the metadata and gradient data using train Scaler...")
        scaled_train_df, train_input_scaler = get_scaled_input_train_data(train_df)
        scaled_val_df = get_scaled_datasets(val_df, train_input_scaler)
        scaled_test_df = get_scaled_datasets(test_df, train_input_scaler)

    print ("Getting the DataLoaders...")
    train_loader, scaler, cc_shape = get_train_dataloader(scaled_train_df, using_moldescs=using_moldescs)
    val_loader = get_val_loader(scaled_val_df, scaler, using_moldescs=using_moldescs)
    test_loader = get_test_loader(scaled_test_df, using_moldescs=using_moldescs)

    print ("Building and training the model...")
    mpnn, trainer = complete_cc_configure_train_model(scaler, train_loader, val_loader, param_dict, cc_shape = cc_shape,results_path=path2res, save_model=True)

    print (f"Writing the results files in {path2res}...")

    test_pred = trainer.predict(mpnn, test_loader)
    test_pred = np.concatenate(test_pred, axis=0)
    res_table = get_res_table(test_df, test_pred, path2res, using_moldescs=using_moldescs)
    mae, rmse, mre, rel_max_error, rel_mean_error = metrics_from_dataframe(res_table)
    write_parameters_file(param_dict, path2res)
    write_metrics_per_cc(res_table, path2res)
    write_metric_txt(mae, rmse, mre, rel_max_error, rel_mean_error, path2res)

    print ("The resuls written successfully! Exiting the program...")
    sys.exit(0)
