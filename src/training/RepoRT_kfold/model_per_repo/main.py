"""
    Performs the k-fold crossvalidation for model_per_repo.

"""
#IMPORT MODULES
import os
import sys
import shutil

from pathlib import Path
from sklearn.model_selection import KFold
from lightning import pytorch as pl
from chemprop import data, nn, models, featurizers

from src.training.functions.splitted_sets_functions import get_res_table, metrics_from_dataframe
from src.training.functions.basic_model_functions import  configure_and_train_mpnn
from src.training.functions.k_fold_functions import *
from src.RepoRT_data_processing.RepoRT_processing import get_processed_df_from_raw

#K-Fold parameters

# K = 10                                                                           # By default, 5 folds will be made.
# ROOT_NAME = "k-fold"
# SIZE_DICT = {f"k-fold{fold_index}":0 for fold_index in range(1, K+1)}           # This will store the size of each split
# OBJECTIVE_DICT = {f"k-fold{fold_index}":[] for fold_index in range(1, K+1)}         # This will store the cc or murcko scaffold
RANDOM_SEED = 42


# DEFINE PARAMETERS
SOURCE_PATH = os.path.join(".", "data", "RepoRT_RP", "processed_data/")                        # This is the source directory that contains all processed files
dataset_type = "no_SMRT"                                                                  # Or with_SMRT, depends on the type of input dataset to use.
apply_grad_down_threshold = False                                                      # Set to True if want to use the filtered by grad_down_threshold
filtering = "filtered" if apply_grad_down_threshold else "no_filtered"
using_moldescs = False                                                                      # Set to True if want to use molecular descriptors for the model
moldesc_dir = "RepoRT_RP_kfold_moldesc" if using_moldescs else "RepoRT_RP_kfold"
path2res = os.path.join(".", "logs", moldesc_dir, dataset_type, filtering, "model_per_repo", "01_23_04_2026/") #Change "dirname" for any name you want.
path2moldesc = os.path.join (".", "data", "with_extra_mol_desc", "extra_mol_descs.tsv")


param_dict = {
    "mp_hidden_dim": 460,                             # Hidden dimension of the message passing (MP) part
    "mp_depth": 4,                                    # Depth/Number of Layers of the MP
    "ffn_hidden_dim": 1400,                            # Hidden layer for the feed-forward network (ffn). This is the regressor
    "ffn_layers": 3,                                  # Number of layers for the ffn.
    "init_lr": 1e-4,                                  # The initial learning rate (lr)
    "max_lr": 1e-3,                                   # Max lr will be reached in after the warm_up epochs.
    "final_lr": 1e-4,                                 # The lr set for the rest of epochs.
    "warm_up_epochs": 2,                              # Number of epochs to reach the max_lr
    "max_epochs": 1000,                               # Set to a smaller number as the datasets here are much smaller.
    "dropout_rate": 0.12,                              # Dropout rate. 0 is default.
    "batch_norm": True,                               # True if want to apply batch_norm
    "metric_list": [nn.MAE(), nn.RMSE()],
    "accelerator": "auto",                            # If GPU and CUDA available change to "gpu". Or can set "cpu" as well.
}



# RUNNING THE SCRIPT
if __name__ == "__main__":
    pl.seed_everything(42, workers=True)

    # Assertion for the data type. Match is used here for better generalization if in the future more dataset types will be evaluated.
    match dataset_type:
        case "no_SMRT":  # These following Booleans are used to get the processed dataset if has not been created yet.
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
            raise NameError(f"Check the dataset_type: {dataset_type}.")

    input_file = os.path.join (path2input, "complete_processed_data.tsv")

    if not Path(input_file).exists():
        get_processed_df_from_raw(drop_smrt=drop_smrt,
                                  down_grad_filter=apply_grad_down_threshold,)

    input_df = pd.read_csv(input_file, sep="\t") #Get the total complete input dataframe

    print("Input data are successfully read. Making the output directory...")
    os.makedirs(path2res, exist_ok=True)

   # MAIN LOOP

    dir_id_array = np.unique(input_df ["dir_id"])

    res_path = os.path.join ("./tmp/")           #Does not really matter, only for temporal saving
    os.makedirs(res_path, exist_ok=True)

    res_dfs_array = []
    metrics_dict = {
        "dir_id": [],
        "k-fold-n":[],
        "n_molecules": [],
        "MAE": [],
        "RMSE": [],
        "MRE": [],
        "rel_max_rt_error": [],
        "rel_mean_rt_error": [],
    }
    for dir_id in dir_id_array:
        temp_df = input_df[input_df["dir_id"] == dir_id]
        kfold_array = []
        temp_df_kf = KFold(n_splits=10,
                           random_state=RANDOM_SEED,
                           shuffle=True)

        for _,fold_index in temp_df_kf.split(temp_df):
            fold_df = temp_df.iloc[fold_index]
            kfold_array.append(fold_df)

        k = len(kfold_array)

        for i in range(k):
            test_df = kfold_array[i]
            val_df = kfold_array[(i + 1) % k]
            train_df = [
                kfold_array[j]
                for j in range(k)
                if j != i and j != (i + 1) % k
            ]
            train_df = pd.concat(train_df, ignore_index=True)

            print ("Getting the DataLoaders...")
            train_loader, scaler = mpr_get_train_loader(train_df)
            val_loader = mpr_get_val_loader(val_df, scaler)
            test_loader = mpr_get_test_loader(test_df)

            print ("Building and training the model...")
            mpnn, trainer = configure_and_train_mpnn(scaler,
                                                     train_loader,
                                                     val_loader,
                                                     param_dict,
                                                     results_path=res_path, #The checkpoints are saved in ./temp/
                                                     save_model=False)

            test_pred = trainer.predict(mpnn, test_loader)
            test_pred = np.concatenate(test_pred, axis=0)
            res_table = get_res_table(test_df, test_pred, res_path, using_moldescs=using_moldescs, save_results=False)
            mae, rmse, mre ,rel_max_error, rel_mean_error = metrics_from_dataframe(res_table)
            # This is extra
            metrics_dict ["dir_id"].append(dir_id)
            metrics_dict ["k-fold-n"].append(f"fold_{i}")
            metrics_dict ["n_molecules"].append(len(test_df))
            metrics_dict["MAE"].append(mae)
            metrics_dict["RMSE"].append (rmse)
            metrics_dict["MRE"].append (mre)
            metrics_dict["rel_max_rt_error"].append(rel_max_error)
            metrics_dict["rel_mean_rt_error"].append(rel_mean_error)

        #Eliminating all ckpt files.
        for ckpt in Path(res_path).glob("*.ckpt"):
            ckpt.unlink()

    #Eliminate the ./temp_dir
    shutil.rmtree(Path(res_path))


    results_df = pd.DataFrame(metrics_dict)
    print ("Writing all the results files.")
    write_model_per_repo_results(results_df, path2res)
    sys.exit(0)
