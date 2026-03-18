"""
Name: RepoRT_MPNN_each_repo.py
Author: Yixi Zhang
Date: March 2026
Version: 1.0
Usege: Builds individuals MPNN using data from every RepoRT's subsets (0001, 0002...) WITH NO MOL DESCs So the input data could be:
    1. "./data/no_extra_mol_desc/RepoRT_only_mol_data.tsv"
    2. "./data/with_extra_mol_desc/RepoRT_only_mol_data.tsv"
In total, there will be 3 main result files saved in the result directory (should be defined):
    1. Metrics.tsv: a tsv file that contains the MAE and RMSE from results of each individual dataset used for training.
    2. Parameters.txt: a txt file containing the parameters used for building the MPNN. It is shared by all repositories.
    3. Results.tsv: a tsv file containing the prediction results for each subset of RepoRT.
"""

# IMPORT THE FUNCTIONS NEEDED

from src.training.functions.basic_model_functions import *
from src.get_raw_data.RepoRT_get_onlymol_data import get_final_dataframe
import sys
import os
from pathlib import Path
# DEFINE THE PARAMETERS. HERE DEFINED ARE THE DEFAULT VALUES OF CHEMPROP
num_repos = 3                                       # This is the number of dataset from RepoRT used for training.
input_path = Path("./data/no_extra_mol_desc/RepoRT_only_mol_data.tsv")    #This is the path to the input data. If not existing, will be created,
path2res = "./logs/RepoRT/model_per_repo/no_mol_desc/Results_0/"
param_dict = {
    "mp_hidden_dim": 300,                             # Hidden dimension of the message passing (MP) part
    "mp_depth": 3,                                    # Depth/Number of Layers of the MP
    "ffn_hidden_dim": 300,                            # Hidden layer for the feed-forward network (ffn). This is the regressor
    "ffn_layers": 1,                                  # Number of layers for the ffn.
    "init_lr": 1e-4,                                  # The initial learning rate (lr)
    "max_lr": 1e-3,                                   # Max lr will be reached in after the warm_up epochs.
    "final_lr": 1e-4,                                 # The lr set for the rest of epochs.
    "warm_up_epochs": 2,                              # Number of epochs to reach the max_lr
    "max_epochs": 1000,                                 # Set to a smaller number as the datasets here are much smaller.
    "dropout_rate": 0,                                # Dropout rate. 0 is default.
    "batch_norm": True,                               # True if want to apply batch_norm
    "metric_list": [nn.MAE(), nn.RMSE()],
    "accelerator": "auto",                            # If GPU and CUDA available change to "gpu". Or can set "cpu" as well.
}

# RUNNING THE SCRIPT
if __name__ == "__main__":
    # Make sure the directory exists.
    os.makedirs(path2res, exist_ok=True)

    #Import the df.
    if input_path.exists():
        print ("The input file is correct!")
    else:
        print ("The file does not exist. Building the file...")
        get_final_dataframe() #If not found, this will build the file from scratch.
    print("Importing the datafile...")
    df = pd.read_csv(input_path, sep='\t')

    #Training process.
    res_array = []
    mae_array = []
    rmse_array = []
    relmax_array = []
    relmean_array = []
    id_array = []
    for id in range (1, num_repos):
        temp_df = df[df["dir_id"] == id]        # This id can be directly used because when imported from tsv file, those "0"s would be eliminated.
        if temp_df.shape [0] != 0:              # This double checks for empty dfs, might be redundant.
            # temp_df = temp_df.sample (50)     #Run this to have a quick test of the Script's usage

            # Building MPNN for each repo
            smiles_array = temp_df.loc [:, "smiles.std"].values
            rts = temp_df.loc [:, ["rt_s"]].values
            scaler, train_loader, val_loader, test_loader, test_indices = get_dataloaders(feature_array=smiles_array, target_array=rts, dataset="RepoRT")
            mpnn, trainer = configure_and_train_mpnn(scaler, train_loader, val_loader, param_dict, path2res, save_model=False)
            test_pred = trainer.predict(mpnn, test_loader)
            test_pred = np.concatenate(test_pred, axis=0)
            # GETTING RESULTS
            res_df = get_res_table_RepoRT (temp_df, test_indices, test_pred)
            mae, rmse, mean_rel_error_max_rt, mean_rel_error_mean_rt = metrics_from_dataframe_RepoRT (res_df)
            res_array.append (res_df)
            mae_array.append (mae)
            relmax_array.append (mean_rel_error_max_rt)
            relmean_array.append (mean_rel_error_mean_rt)
            rmse_array.append (rmse)
            id_array.append (id)
        else:
            print (f"The repo nº {id} does not contain any data. It will be skipped.")
            continue

    # HERE THE OVERALL RESULTS ARE WRITTEN INTO A SINGLE FILE
    print ("Writing the parameters' file...")
    write_parameters_file(param_dict, path2res)
    # Get the result table.
    final_res_table = pd.concat (res_array)
    filename = path2res + "Results_table.tsv"
    print ("Writing the Results table...")
    final_res_table.to_csv (filename, index = False)
    print ("Writing the metrics' file...")
    write_metric_tsv(id_array, mae_array, rmse_array, relmax_array, relmean_array, path2res)
    print ("All files successfully written !")
    sys.exit(0)




