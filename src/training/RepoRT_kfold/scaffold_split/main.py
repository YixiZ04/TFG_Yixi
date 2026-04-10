#IMPORT MODULES
import os
import sys
from pathlib import Path
from lightning import pytorch as pl
from src.training.functions.splitted_sets_functions import *
from src.training.functions.k_fold_functions import split_dataset_into_k_folds


#K-Fold parameters

K = 10                                                                           # By default, 5 folds will be made.
ROOT_NAME = "k-fold"
SIZE_DICT = {f"k-fold{fold_index}":0 for fold_index in range(1, K+1)}           # This will store the size of each split
OBJECTIVE_DICT = {f"k-fold{fold_index}":[] for fold_index in range(1, K+1)}         # This will store the cc or murcko scaffold


# DEFINE PARAMETERS

dataset_type = "no_SMRT" #Or with_SMRT, depends on the type of input dataset to use.
using_moldescs = False     # Set to True if want to use molecular descriptors for the model
moldesc_dir = "RepoRT_kfold_moldesc" if using_moldescs else "RepoRT_kfold"
path2res = os.path.join(".", "logs", moldesc_dir, dataset_type, "scaffold_split", "dirname/")
path2moldesc = os.path.join (".", "data", "with_extra_mol_desc", "extra_mol_descs.tsv")
param_dict = {
    "mp_hidden_dim": 300,                             # Hidden dimension of the message passing (MP) part
    "mp_depth": 3,                                    # Depth/Number of Layers of the MP
    "ffn_hidden_dim": 300,                            # Hidden layer for the feed-forward network (ffn). This is the regressor
    "ffn_layers": 1,                                  # Number of layers for the ffn.
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
            apply_upthreshold = False
            processed_filename = "no_SMRT_no_ds_data.tsv" #filename for the processed .tsv file.
        case "with_SMRT":
            drop_smrt = False
            apply_upthreshold = True
            processed_filename = "with_SMRT_ds_data.tsv"
        case _:
            raise NameError (f"Check the dataset_type: {dataset_type}.")

    # Define the processed file according to the dataset type given.
    split_path = os.path.join (".", "data", "processed_RepoRT", dataset_type, "scaffold_split_data", "ms_complete_data.tsv")
    input_df = pd.read_csv(split_path, sep="\t")

    print("Input data are successfully read. Making the output directory...")
    os.makedirs(path2res, exist_ok=True)

    kfold_array = split_dataset_into_k_folds(input_df, OBJECTIVE_DICT, SIZE_DICT, "ms_smiles")
    k = len(kfold_array)

    for i in range(k):
        res_path = os.path.join (path2res, f"kfold_{i}/")
        test_df = kfold_array[i]
        val_df = kfold_array[(i + 1) % k]
        train_df = [
            kfold_array[j]
            for j in range(k)
            if j != i and j != (i + 1) % k
        ]
        train_df = pd.concat(train_df, ignore_index=True)
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
        val_loader = get_test_val_loader(scaled_val_df, scaler, using_moldescs=using_moldescs)
        test_loader = get_test_val_loader(scaled_test_df, scaler, using_moldescs=using_moldescs)

        print ("Building and training the model...")
        mpnn, trainer = complete_cc_configure_train_model(scaler, train_loader, val_loader, param_dict, cc_shape = cc_shape,results_path=res_path, save_model=True)

        print (f"Writing the results files in {res_path}...")

        test_pred = trainer.predict(mpnn, test_loader)
        test_pred = np.concatenate(test_pred, axis=0)
        res_table = get_res_table(test_df, test_pred, res_path, using_moldescs=using_moldescs)
        mae, rmse, rel_max_error, rel_mean_error = metrics_from_dataframe(res_table)
        write_parameters_file(param_dict, res_path)
        write_metrics_per_cc(res_table, res_path)
        write_metric_txt(mae, rmse, rel_max_error, rel_mean_error, res_path)

        print ("The resuls written successfully! Exiting the program...")
