"""
Name: perform_cc_scaffold_split.py
Author: Yixi Zhang
Date: March 2026
Version: 1.2.
Description: Contains functions for performing splitting by both chromatography condition and Murcko Scaffold, this is, there should not be molecules of, for example,
directory 0001 and murcko-scaffold ccccccc1N in different datasets. This is to avoid data leakage and test if a MPNN can really be projectable.
Description: Based on RDkit and Bemis-Murcko Scaffold to avoid data leakage, this is, to make sure each dataset (train/test/val) has different scaffolds.
Update (1.1.): Implemented the new version of data_processing.py with the filtered datafile.
Update (1.2.): Adapted to the version 1.2. of data_processing.py, as now this splitting function is more flexible.
"""

# IMPORT MODULES
from sklearn.model_selection import train_test_split
from src.training.RepoRT.scaffold_split.perform_scaffold_split import *

def cc_ms_split (ms_complete_file, save_dir, random_seed, processed_file, save_complete_ms_dir, apply_upthreshold, drop_smrt):
    """
    This functions obtains splitted datafiles by both chromatography condition and Murcko Scaffold.
    The main input file is the ms_complete_file, if not exist, it will be created using ms_split().
    The random seed = 51 just works for this version, the difference between val and test datasets is not too much.
    NOTE: This makes sure that the val and test set both have similar dimensions and makes this splitting to be automatic.
    """
    # Check the input files.
    if not Path(ms_complete_file).exists():
        print ("Creating the Scaffold split file...")
        ms_split(input_path= processed_file,
                 output_dir=save_complete_ms_dir,
                 apply_upthreshold=apply_upthreshold,
                 drop_smrt=drop_smrt)

    print ("Reading the input file...")
    input_df = pd.read_csv (ms_complete_file, sep="\t")
    index_array = np.unique (input_df ["dir_id"]) #This ensures that it contains all repo indices.

    print ("Checking for the output dir...")
    os.makedirs(save_dir, exist_ok=True)

    print ("Splitting by ms_smiles the input data...")
    #First splitting by ms_smiles
    ms_smiles_array = np.unique (input_df ["ms_smiles"])
    np.random.seed (random_seed)
    np.random.shuffle (ms_smiles_array)
    train_ms_smiles, val_ms_smiles, test_ms_smiles = [],[],[]
    train_size = val_size = test_size = 0
    threshold = round (input_df.shape [0]/3) #Make sure that there are approx. 1/3 molecules in the first split.
    for ms_smiles in ms_smiles_array:
        temp_df = input_df[input_df["ms_smiles"] == ms_smiles]
        if train_size + temp_df.shape [0]< threshold:
            train_ms_smiles.append(ms_smiles)
            train_size += temp_df.shape[0]
        elif val_size + temp_df.shape [0]< threshold:
            val_ms_smiles.append (ms_smiles)
            val_size += temp_df.shape[0]
        else:
            test_ms_smiles.append(ms_smiles)
            test_size += temp_df.shape[0]
    train_ms_split_df = input_df[input_df["ms_smiles"].isin(train_ms_smiles)]
    val_ms_split_df = input_df[input_df["ms_smiles"].isin (val_ms_smiles)]
    test_ms_split_df = input_df[input_df["ms_smiles"].isin (test_ms_smiles)]

    print("Performing second splitting by dir_ids...")
    # Perform dir_id splitting
    np.random.shuffle (index_array)
    train_indices, test_indices = train_test_split (index_array, test_size=0.1, random_state=42)
    train_indices, val_indices = train_test_split (train_indices, test_size=0.1111, random_state=42)
    final_train_df = train_ms_split_df[train_ms_split_df["dir_id"].isin (train_indices)]
    final_val_df = val_ms_split_df[val_ms_split_df["dir_id"].isin (val_indices)]
    final_test_df = test_ms_split_df[test_ms_split_df["dir_id"].isin (test_indices)]

    print (f"Saving result files to {save_dir}")
    final_train_df.to_csv (f"{save_dir}train_data.tsv", sep='\t', index=False)
    final_val_df.to_csv (f"{save_dir}val_data.tsv", sep='\t', index=False)
    final_test_df.to_csv(f"{save_dir}test_data.tsv", sep='\t', index=False)
    return




