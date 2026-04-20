"""
    Name: data_processing.py
    Author: Yixi Zhang
    Version: 1.2.
    Description: This performsthe processing on preprocessed RepoRT data. This process is done as shown in the following:
        1. Eliminates columns that are not useful. Such as comment, classyfires columns.
        (The molecule name is not eliminated as this can be useful for later to identify the molecules easier).
        2. Drop those molecules having > 10 gradients and  those having < 3 segments
        3. Drop those repos that contains < 100 molecules.
        3. Get max and mean RT for each repository, merge them to each molecule according the dir_id.
    Update: Rectified a concept error here, as the final result table should not be scaled, and the scaling process will be done after data splitting using train set Scaler
    Also, on top of the previous update, here more options are given:
        1. Drop completely the SMRT dataset and apply down_threshold. (Drop datasets if contains less molecules than the threshold).
        2. Not dropping completely the SMRT and apply a up_threshold, meaning that we are performing a downsampling process.
    Overall, if all options were tested, this file could produce 6 different tsv files in ./data/processed_RepoRT/ and identified by their filename:
        1. no_SMRT_complete_data.tsv (Dropped SMRT but no any filter has been applied, this is, it contains repos < 100 molecules)
        2. no_SMRT_ds_data.tsv (Dropped SMRT and applied downsampling (ds), the repos contains 100 < n < 5000 molecules)
        3. no_SMRT_no_ds_data.tsv (Dropped SMRT and applied ds, the repos contains n > 100 molecules.
        4. with_SMRT_complete_data.tsv (Not dropped SMRT and repos contains any number of molecules).
        5. with_SMRT_ds_data.tsv. This would be the "filtered_treated_data.tsv" in the previous version applying the filters.
        6. with_SMRT_no_ds_data.tsv. Containing SMRT and no ds applied, but all repos contains n > 100 molecules.
    NOTE: This is possible to create all 6 files, but it does not mean that we will evaluate models on all of them.
"""
# IMPORT MODULES AND SCRIPTS

import os

import pandas as pd
import numpy as np

from pathlib import Path

from src.RepoRT_data_processing.RepoRT_preprocessing import get_preprocessed_datatable


# VARIABLES

SOURCE_PATH = os.path.join (".", "data", "RepoRT_RP", "processed_data/")
PATH2INPUTS = os.path.join(".", "data", "RepoRT_RP", "preprocessed_data/")
RT_INPUT = os.path.join(PATH2INPUTS, "preprocessed_rt_data.tsv")
CC_INPUT = os.path.join(PATH2INPUTS, "preprocessed_cc_data.tsv")
GRAD_INPUT = os.path.join(PATH2INPUTS, "preprocessed_gradient_data.tsv")
GRAD2DROP_UP = 11
GRAD2DROP_DOWN = 2
MOL_FILTER_DOWN = 100
MOL_FILTER_UP = 5000


#HELPER FUNCTIONS

def _get_input_df (rt_input = RT_INPUT,
                   cc_input = CC_INPUT,
                   grad_input = GRAD_INPUT):
    """
        This function checks for the input files and loads then as pd.DataFrames
    """
    print ("Cheking for the input files...")
    if (not Path(rt_input).exists() or
        not Path(cc_input).exists() or
        not Path(grad_input).exists()):
        get_preprocessed_datatable()
    else:
        print ("The input files exists!")

    print ("Fetching the input tables...")
    rt_df = pd.read_csv(rt_input, sep='\t', dtype={"dir_id":str})
    cc_df = pd.read_csv(cc_input, sep='\t', dtype={"dir_id":str})
    grad_df = pd.read_csv(grad_input, sep='\t', dtype={"dir_id":str})

    return rt_df, cc_df, grad_df

def _write_rt_dropped_columns_report (dropped_columns, path2dir, filename):
    """
        This function writes a Report with all the dropped columns
    """
    path2file = os.path.join(path2dir, "report_files",filename)
    with open(path2file, "w") as f:
        f.write("These columns have been eliminated from the retention time data:\n")
        f.writelines(f"{column}\n" for column in dropped_columns)

def _write_down_filtering_report(dropped_array,path2dir, filename, down_threshold = MOL_FILTER_DOWN):
    """
        Writes a file in the directory indicated with information of the dropped molecules for containing <threshold moelcules
    """
    path2file = os.path.join (path2dir, "report_files",filename)
    with open(path2file, "w") as f:
        f.write(f"These repositories have been eliminated for containing less than {down_threshold} molecules:\n")
        f.writelines(f"{index}: {n_mols} molecules\n" for dictionary in dropped_array for index, n_mols in dictionary.items())

def _write_downsampling_report(ds_array, path2dir, filename, up_threshold=MOL_FILTER_UP):
    """
        Writes a report file containing the information of downsampled repos
    """
    path2file = os.path.join(path2dir, "report_files",filename)
    with open(path2file, "w") as f:
        f.write(f"These repositories have been downsampled for containing more than {up_threshold} molecules:\n")
        f.writelines(
            f"{index}: {n_mols} molecules\n" for dictionary in ds_array for index, n_mols in dictionary.items())

def _write_dropped_cc_columns_report (dropped_columns_array, path2dir, filename):
    """
        Writes a report file containing the information of the dropped columns of the cc data
    """
    path2file = os.path.join (path2dir, "report_files",filename)
    with open(path2file, "w") as f:
        f.write(f"The columns have been removed from the cc data as they have no purpose for training models:\n")
        f.writelines (f"{index}\n" for index in dropped_columns_array)

def _write_dropped_grad_down_filter_report(dropped_grad_array, path2dir, filename, down_threshold = GRAD2DROP_DOWN):
    """ 
        Writes a report file containing information of which repos have been dropped because of containing less than down_threshold segments
    """
    path2file = os.path.join(path2dir, "report_files", filename)
    with open(path2file, "w") as f:
        f.write(f"These repositories have been eliminated for its gradient data contains less than {down_threshold + 1} segments:\n")
        f.writelines(f"{index}\n" for index in dropped_grad_array)

def _write_dropped_grad_up_filter_report(dropped_grad_array, path2dir, filename, up_threshold = GRAD2DROP_UP):
    """
        Writes a report file containing information of which repos have been dropped because of containing less than down_threshold segments
    """
    path2file = os.path.join(path2dir,"report_files", filename)
    with open(path2file, "w") as f:
        f.write(f"These repositories have been eliminated for its gradient data contains more than {up_threshold + 1} segments:\n")
        f.writelines(f"{index}\n" for index in dropped_grad_array)

def _write_complete_info_report(complete_df, path2dir, filename):
    """
        This function writes an report containing 2 informations:
            - The molecule count.
            - The directories contained
    """
    n_molecules = len(complete_df)
    repos = complete_df["dir_id"].unique()
    size_dict = {}
    for repo in repos:
        temp_df = complete_df[complete_df["dir_id"] == repo]
        size_dict[repo] = len(temp_df)

    path2file = os.path.join(path2dir, "report_files",filename)
    with open(path2file, "w") as f:
        f.write(f"This dataset contains {n_molecules} molecules\n")
        f.write(f"The directory contained in this dataset are:\n")
        f.writelines(f"{index}: {size}\n" for index, size in size_dict.items())

# DEFINE FUNCTIONS
def _drop_rt_columns (rt_df,
                      path2dir):
    """
        Input: preprocessed df from RepoRT.
        Output: Updated df with only useful columns, dropped columns:
        ["comment", "inchikey.std", all columns containing "classyfire."]
    """
    columns_to_drop = ["comment", "inchikey.std"]
    for column in rt_df.columns:
        if "classyfire" in column:
            columns_to_drop.append(column)
    df = rt_df.drop (columns = columns_to_drop)

    _write_rt_dropped_columns_report(columns_to_drop, path2dir, "Report_rt_data_dropped_columns.txt")
    return df

def _filer_by_down_threshold (rt_df,
                              path2dir,
                              threshold = MOL_FILTER_DOWN):
    """
        This function eliminates those repos containing less than the threshold molecules, which is set to 100 molecules as default
    """

    index_array = np.unique(rt_df["dir_id"])
    final_df_array, dropped_array = [], []

    for index in index_array:
        temp_df = rt_df[rt_df["dir_id"]==index]
        if len(temp_df) >= threshold:
            final_df_array.append(temp_df)
        else:
            dropped_array.append({index:len(temp_df)})
    _write_down_filtering_report(dropped_array,
                                 path2dir,
                                 "Report_down_filetering.txt")

    final_df = pd.concat(final_df_array, ignore_index=True)

    return final_df


def _downsample(rt_df,
                path2dir,
                up_threshold = MOL_FILTER_UP,
                ):
    """
        This function performs downsampling on RepoRT data and writes a report file for it.
        As these should be applied after filtering by down_threshold, all the repo < up_threshold contains >100 molecules
    """
    index_array =  rt_df["dir_id"].unique()
    final_df, ds_repos = [], []
    for index in index_array :
        temp_df = rt_df[rt_df["dir_id"] == index]
        if  temp_df.shape[0] < up_threshold:
            final_df.append (temp_df)
        else:
            final_df.append (temp_df.sample (up_threshold, random_state=42))
            ds_repos.append ({index: len(temp_df)})

    _write_down_filtering_report(ds_repos, path2dir, "Report_downsampling.txt")

    return pd.concat (final_df)

def _find_doublets(rt_df, path2dir):
    """
        Retrieves all the doublet entrees of RepoRT and the dataset we are using for training models.
        Save the duplicated values as a .tsv in the directory where processed RepoRT data is found.
    """
    doublets_df = rt_df [rt_df.duplicated(subset=["dir_id", "smiles.std"], keep=False)]
    path2file = os.path.join(path2dir, "doublets.tsv")
    doublets_df.to_csv(path2file, sep="\t", index=False)

def _get_max_mean_rt_per_cc (rt_df):
    """
        This function get the max and mean rt for every chromatography column and inserts them next to "rt_s" column of the dataframe.
        Updates the input_dataframe.
    """
    max_array, mean_array = [], []
    index_array = np.unique (rt_df["dir_id"])
    for index in index_array:
        temp_df = rt_df [rt_df ["dir_id"] == index]
        mean_rt = np.mean (temp_df["rt"])
        max_rt = np.max (temp_df["rt"])
        temp_max_array = [ max_rt for _ in range (temp_df.shape[0])]
        temp_mean_array = [ mean_rt for _ in range (temp_df.shape [0])]
        max_array = max_array + temp_max_array
        mean_array = mean_array + temp_mean_array

    position = rt_df.columns.get_loc ("rt")
    rt_df.insert (position + 1, "max_rt", max_array)
    rt_df.insert (position + 2, "mean_rt", mean_array)

def _get_processed_rt_df (rt_df,
                          downsampling,
                          path2dir):
    """
        This function processed the preprocessed_rt_data
    """
    print ("Processing Retention time data...")
    rt_df = _drop_rt_columns(rt_df,path2dir)
    rt_df = _filer_by_down_threshold(rt_df, path2dir)
    if downsampling:
        rt_df = _downsample(rt_df, path2dir)
    else:
        print("Not applting downsampling.")
    _find_doublets(rt_df, path2dir)
    _get_max_mean_rt_per_cc(rt_df)

    path2file = os.path.join(path2dir, "processed_rt_data.tsv")
    rt_df.to_csv(path2file, sep="\t", index=False)

    return rt_df

# DEFINE CC PROCESSING FUNCTION

def _drop_cc_columns (cc_df,
                      path2dir):
    """
        This functions drops those columns not used in the cc data:
        [column.name, column.usp.code], the last one is because it has already been OneHotEncoded, there is no purpose to keep the original column.
    """
    drop_columns = ["column.name", "column.usp.code"]
    final_df = cc_df.drop (columns = drop_columns)

    _write_dropped_cc_columns_report(drop_columns, path2dir, "Report_dropped_cc_columns.txt")
    path2file = os.path.join(path2dir, "processed_cc_data.tsv")
    final_df.to_csv(path2file, sep="\t", index=False)

    return final_df


# DEFINE GRADIENT PROCESSING FUNCTION

def _drop_grad_data_by_down_threshold(grad_df,
                                      path2dir,
                                      down_threshold = GRAD2DROP_DOWN
                                      ):
    """
        This functions drops those repositories for containing less than the down_threshold +   gradient segment.
        The grad_df has not be filled with 0, nan values are kept
    """
    threshold_column = f"t [min]_{down_threshold}"
    final_df = grad_df[grad_df[threshold_column].notna()]
    dropped_repos = grad_df.loc [~grad_df["dir_id"].isin(final_df["dir_id"]), "dir_id"].values

    _write_dropped_grad_down_filter_report(dropped_repos,
                                           path2dir,
                                           "Report_downfiltering.txt")

    return final_df

def _drop_grad_data_by_up_threshold (grad_df,
                                     path2dir,
                                     up_threshold=GRAD2DROP_UP):
    """
        This function eliminates all the repositories whose gradient has more than up_threshold + 1 segments.
        To be technical, as Python initializes indexes with 0, those repositories.
        Remember here the NaN values still are there, not filled with 0 yet.
    """
    threshold_column = f"t [min]_{up_threshold}"
    previous_column = f"D [%]_{up_threshold-1}"
    final_df = grad_df[grad_df[threshold_column].isna()].loc[:, :previous_column]
    dropped_repos = grad_df.loc [~grad_df["dir_id"].isin(final_df["dir_id"]), "dir_id"].values
    _write_dropped_grad_up_filter_report(dropped_repos,
                                         path2dir,
                                         "Report_upfiltering.txt")

    return final_df


def _get_processed_grad_df (grad_df,
                            path2dir,
                            apply_down_filtering):
    """
        This function summarizes all the action of the 2 previous functions, so only this need to be called when processing the gradient_data.
        There is a Boolean used to decide whether to apply or not the down filtering for the gradient data.
    """

    print("Processing gradient data...")

    if apply_down_filtering:
        grad_df = _drop_grad_data_by_down_threshold(grad_df, path2dir)
        grad_df = _drop_grad_data_by_up_threshold(grad_df, path2dir)
    else:
        grad_df = _drop_grad_data_by_up_threshold(grad_df, path2dir)

    grad_df = grad_df.fillna(0)

    path2file = os.path.join(path2dir, "processed_grad_data.tsv")
    grad_df.to_csv(path2file, sep="\t", index=False)

    return grad_df


# COMPLETE PROCESSED DATA
def _get_complete_processed_data (rt_df, cc_df, grad_df, path2dir):
    """
        This functions performs an inner join to all three dataframes and saves it in the directory indicated
    """

    rt_cc_df = pd.merge (rt_df, cc_df, on = "dir_id", how = "inner")
    final_df = pd.merge (rt_cc_df, grad_df, on = "dir_id", how = "inner")

    path2file = os.path.join(path2dir, "complete_processed_data.tsv")
    final_df.to_csv(path2file, sep="\t", index=False)

    return final_df



# Main function

def get_processed_df_from_raw (source_path = SOURCE_PATH,
                               drop_smrt = True,
                               down_grad_filter = False,
                               smrt_id = "0186",
                               ):
    """
        This function will build an entire directory containing the rt data, cc data, grad data and all the report files,
        Here, 4 directories can be built depending on the Booleans' value:
            - Containing SMRT but passed through a downsampling process.
            - Not containing SMRT. No downsampling is performed.
            - Containing SMRT and downsampled. But also eliminated those repositories with less than down_grad_filter segments.
            - Not containing SMRT and NO downsampling. But also eliminated those repositories with less than down_grad_filter segments.
        The up_grad_filter and down_mol_filter will be applied to ALL cases.
    """
    rt_df, cc_df, grad_df = _get_input_df()

    print ("Cheking for the output directory...")
    if drop_smrt:
        downsampling = False
        rt_df = rt_df[rt_df["dir_id"] != smrt_id]
        if down_grad_filter:
            path2dir = os.path.join(source_path, "no_SMRT_down_grad_filter/")
        else:
            path2dir = os.path.join(source_path, "no_SMRT/")
    else:
        downsampling = True
        if down_grad_filter:
            path2dir = os.path.join(source_path, "with_SMRT_down_grad_filter/")
        else:
            path2dir = os.path.join(source_path, "with_SMRT/")

    print ("Making the directory...")
    os.makedirs(path2dir, exist_ok = True)
    os.makedirs(os.path.join(path2dir, "report_files/"), exist_ok = True)

    print ("Getting processed rt data...")
    processed_rt_df = _get_processed_rt_df(rt_df,
                                           path2dir=path2dir,
                                           downsampling=downsampling)

    print ("Getting processed cc data...")
    processed_cc_df  =_drop_cc_columns(cc_df, path2dir)

    print ("Getting processed grad_data...")
    processed_grad_df = _get_processed_grad_df(grad_df,
                                               path2dir=path2dir,
                                               apply_down_filtering=down_grad_filter)
    print ("Getting the complete processed data...")
    complete_df = _get_complete_processed_data(processed_rt_df,
                                 processed_cc_df,
                                 processed_grad_df,
                                 path2dir)
    _write_complete_info_report(complete_df, path2dir, "Report_complete_info.txt")
    print (f"All the file saved in {path2dir}!!")
    return

if __name__ == "__main__":
    get_processed_df_from_raw(drop_smrt=False,
                              down_grad_filter=False,)