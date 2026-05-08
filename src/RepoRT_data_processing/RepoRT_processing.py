"""
    Name: RepoRT_processing.py
    Author: Yixi Zhang
    Description:
    This includes a processing for duplicated chromatography conditions. Those who share the same cc is unified into another unique id;
    and all the processing is done using the new unique ID.
    NOTE: The repeated molecules are considered as "doublets" within the new id.
    This performs the processing on preprocessed RepoRT data:
        1. Eliminates columns that are not useful. Such as comment, classyfires columns.
        (The molecule name is not eliminated as it can be useful for later to identify the molecules easier).
        2. Drop those molecules having > 12 segments gradients and  those having < 3 segments
        3. Drop those repos that contains < 100 molecules.
        3. Get max and mean RT for each repository, merge them to each molecule according the dir_id (**cc_id)
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
from src.RepoRT_data_processing.RepoRT_get_raw_data import merge_complete_file


# VARIABLES

SOURCE_PATH = os.path.join (".", "data", "RepoRT_RP", "processed_data/")
PATH2INPUTS = os.path.join(".", "data", "RepoRT_RP", "preprocessed_data/")
PATH2RAW_RT = os.path.join (".", "data", "RepoRT_RP", "raw_data", "raw_rt_data.tsv")
PATH2RAW_CC = os.path.join (".", "data", "RepoRT_RP", "raw_data", "raw_cc_data.tsv")
PATH2RAW_GRAD = os.path.join (".", "data", "RepoRT_RP", "raw_data", "raw_grad_data.tsv")
PATH2RT= os.path.join(PATH2INPUTS, "preprocessed_rt_data.tsv")
PATH2CC = os.path.join(PATH2INPUTS, "preprocessed_cc_data.tsv")
PATH2GRAD = os.path.join(PATH2INPUTS, "preprocessed_gradient_data.tsv")
DROP_REPO = ["0093", "0150", "0434", "0435"]
DOUBLET_THRESHOLD = 0.05
GRAD2DROP_UP = 11
GRAD2DROP_DOWN = 2
MOL_FILTER_DOWN = 100
MOL_FILTER_UP = 5000


#HELPER FUNCTIONS

def _get_raw_input_df (rt_input,
                       cc_input,
                       grad_input):
    """
        To fetch the raw df for duplicated cc check. Otherwise, it does not work...
    """
    print("Cheking for the input files...")
    if (not Path(rt_input).exists() or
        not Path(cc_input).exists() or
        not Path(grad_input).exists()):
        merge_complete_file()
    else:
        print("The input files exists!")

    print("Fetching the input tables...")
    rt_df = pd.read_csv(rt_input, sep='\t', dtype={"dir_id": str})
    cc_df = pd.read_csv(cc_input, sep='\t', dtype={"dir_id": str}).fillna(-1)
    grad_df = pd.read_csv(grad_input, sep='\t', dtype={"dir_id": str}).fillna(0)

    return rt_df, cc_df, grad_df

def _get_input_df (rt_input = PATH2RT,
                   cc_input = PATH2CC,
                   grad_input = PATH2GRAD):
    """
        This function checks for the input files and loads then as pd.DataFrames.
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
        This function writes a report containing 2 information:
            - The molecule count.
            - The directories contained
    """
    n_molecules = len(complete_df)
    repos = complete_df["cc_id"].unique()
    size_dict = {}
    for repo in repos:
        temp_df = complete_df[complete_df["cc_id"] == repo]
        size_dict[repo] = len(temp_df)

    path2file = os.path.join(path2dir, "report_files",filename)
    with open(path2file, "w") as f:
        f.write(f"This dataset contains {n_molecules} molecules\n")
        f.write(f"The directory contained in this dataset are:\n")
        f.writelines(f"{index}: {size}\n" for index, size in size_dict.items())


## Check Duplicated cc, and get the dir_ids array

def _get_duplicated_entrees(df, droppped_repo) -> pd.DataFrame:
    """
        Gets the duplicated entrees from a dataframe.
        Drops some dir_ids for inconsistencies.
    """

    temp_df = df.copy()
    duplicated_df = temp_df[temp_df.duplicated(subset=temp_df.columns[1:], keep=False)]
    final_df = duplicated_df[~duplicated_df["dir_id"].isin(droppped_repo)]

    return final_df


def _group_up_by_conditions(df) -> list:
    """
       Gets an array contains dataframes for the same chromatography conditions.
       Output: Array of dataframes
    """
    columns2use = list(df.columns[1:])

    grouped_df = df.groupby(by=columns2use)
    final_array = []
    for _, same_cc_df in grouped_df:
        final_array.append(same_cc_df)

    return final_array


def _get_repeated_condition_df(df_array) -> pd.DataFrame:
    """
        Saves a .tsv file containing the repeated conditions + dir_ids
    """
    temp_dict = {
        "repeated_conditions": [],
        "dir_ids": [],
    }
    for index, df in enumerate(df_array):
        temp_dict["repeated_conditions"].append(index)
        temp_dict["dir_ids"].append(df.loc[:, "dir_id"].values)

    final_df = pd.DataFrame(temp_dict)

    return final_df


def _get_duplicated_dirids (cc_grad_df,
                            dropped_repo) -> list:
    """
        Get a 2D array containing the dir_ids for a same cc.
    """

    duplicated_ccs = _get_duplicated_entrees(cc_grad_df, dropped_repo)
    grouped_array = _group_up_by_conditions(duplicated_ccs)

    final_df = _get_repeated_condition_df(grouped_array)
    duplicated_dirid_array = list(final_df.loc[:, "dir_ids"].values)
    return duplicated_dirid_array

def _flat_2d_array (array) -> list:
    """
        Flattens a 2D array into a 1D array.
        This does list by list, not row-wise.
    """
    return [item for sublist in array for item in sublist]

def _get_all_dirid (df,
                    duplicated_dirids_array) -> list:
    """
        Make the final 2D array containing all the repeated and not repeated array
    """
    flat_array = _flat_2d_array(duplicated_dirids_array)
    dir_id_array = np.unique (df.loc[:,"dir_id"].values)

    for dir_id in dir_id_array:
        if dir_id not in flat_array:
            duplicated_dirids_array.append([dir_id])

    return list(duplicated_dirids_array)

def _merge_same_cc_rt (df,
                       duplicated_dirids_array,
                       drop_repos) -> tuple [pd.DataFrame, pd.DataFrame]:
    """
        Eliminates the dir_id column and substitute it by cc_id.
        Also returns another df indicating the corresponding dir_id to a cc_id.
        This last df is considered to be the mapping df, will be used later to ensure the cc_id is
        consistent in both cc_df and grad_df
    """
    temp_array = []
    temp_dict = {"cc_id":[],
                 "dir_ids":[]}
    dir_ids_array = _get_all_dirid(df,
                                   duplicated_dirids_array)

    for index, dir_ids in enumerate(dir_ids_array):
        temp_df = df[df["dir_id"].isin(dir_ids)]
        if len(temp_df) > 0:
            temp_cc_id_array = [ f"cc_{index}" for _ in range(len(temp_df))]
            temp_df.insert(0, "cc_id", temp_cc_id_array)
            temp_array.append(temp_df)
            temp_dict ["cc_id"].append(f"cc_{index}")
            temp_dict ["dir_ids"].append(list(dir_ids))

    temp_df = pd.concat(temp_array, ignore_index=True)
    final_df = temp_df[~temp_df ["dir_id"].isin(drop_repos)].copy()
    final_df.drop("dir_id", axis=1, inplace=True)

    return final_df, pd.DataFrame(temp_dict)

def _map_cc_id2df (df,
                   mapping_df,
                   drop_repos) -> pd.DataFrame:
    """
        To ensure the same cc_id on the same original dir_ids, the df obtained with the previous function
        is used for this.
    """

    exploded_df = mapping_df.explode("dir_ids") # Needed as it is a list

    temp_df = pd.merge (df,
                        exploded_df,
                        left_on="dir_id",
                        right_on="dir_ids",
                        how="inner")

    temp_df.drop_duplicates(subset="cc_id", inplace=True)
    final_df = temp_df[~temp_df["dir_id"].isin(drop_repos)].copy()

    final_df.drop(columns=["dir_id", "dir_ids"],  inplace=True)
    column2move = final_df.pop("cc_id")
    final_df.insert(0, "cc_id", column2move)


    return final_df


def _duplicated_cc_main (path2raw_rt,
                         path2raw_cc,
                         path2raw_grad,
                         rt_df,
                         cc_df,
                         grad_df,
                         drop_repos,
                         path2res) -> tuple [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
        Need raw datatables as inputs, otherwise, the duplicated cc are not checked.
        This function performs the very first processing for the data: group up the data with the same cc
        and substitutes them with a new unique id: "cc_id"
        Also a .tsv file is saved in the path indicated to see the corresponding dir_ids for a cc_id.
        The output dfs of this function are used for later processing.
    """
    raw_rt_df, raw_cc_df, raw_grad_df = _get_raw_input_df(path2raw_rt,
                                                          path2raw_cc,
                                                          path2raw_grad)
    cc_grad_df = pd.merge (raw_cc_df,
                           raw_grad_df,
                           on="dir_id",
                           how="inner")

    duplicated_dirids = _get_duplicated_dirids(cc_grad_df,drop_repos)

    non_duplicated_rt_df,cc_id_dirid_df  = _merge_same_cc_rt(rt_df, duplicated_dirids,drop_repos)
    non_duplicated_cc_df = _map_cc_id2df(cc_df, cc_id_dirid_df,drop_repos)
    non_duplicated_grad_df = _map_cc_id2df(grad_df, cc_id_dirid_df,drop_repos)

    path2file = os.path.join (path2res, "cc_id_vs_dir_id.tsv")
    cc_id_dirid_df.to_csv(path2file,sep='\t', index=False)

    return non_duplicated_rt_df, non_duplicated_cc_df, non_duplicated_grad_df


# PROCESS RT DATA
def _drop_rt_columns (rt_df,
                      path2dir):
    """
        Input: preprocessed df from RepoRT.
        Output: Updated df with only useful columns, dropped columns:
        ["comment", "inchikey.std", all columns containing "classyfire"]
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

    index_array = np.unique(rt_df["cc_id"])
    final_df_array, dropped_array = [], []

    for index in index_array:
        temp_df = rt_df[rt_df["cc_id"]==index]
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
    index_array =  rt_df["cc_id"].unique()
    final_df, ds_repos = [], []
    for index in index_array :
        temp_df = rt_df[rt_df["cc_id"] == index]
        if  temp_df.shape[0] < up_threshold:
            final_df.append (temp_df)
        else:
            final_df.append (temp_df.sample (up_threshold, random_state=42))
            ds_repos.append ({index: len(temp_df)})

    _write_down_filtering_report(ds_repos, path2dir, "Report_downsampling.txt")

    return pd.concat (final_df)

def _get_max_mean_rt_per_cc (rt_df):
    """
        This function get the max and mean rt for every chromatography column and inserts them next to "rt_s" column of the dataframe.
        Updates the input_dataframe.
    """
    max_array, mean_array = [], []
    index_array = np.unique (rt_df["cc_id"])
    for index in index_array:
        temp_df = rt_df [rt_df ["cc_id"] == index]
        mean_rt = np.mean (temp_df["rt"])
        max_rt = np.max (temp_df["rt"])
        temp_max_array = [ max_rt for _ in range (temp_df.shape[0])]
        temp_mean_array = [ mean_rt for _ in range (temp_df.shape [0])]
        max_array = max_array + temp_max_array
        mean_array = mean_array + temp_mean_array

    position = rt_df.columns.get_loc ("rt")
    rt_df.insert (position + 1, "max_rt", max_array)
    rt_df.insert (position + 2, "mean_rt", mean_array)
#Doublets treating functions

def _find_doublets(rt_df, path2dir):
    """
        Retrieves all the doublet entrees of RepoRT and the dataset we are using for training models.
        Save the duplicated values as a .tsv in the directory where processed RepoRT data is found.
        This function outputs 2 Dataframes (while mantaining the original one untouched):
            - A df without any doublets at all
            - A df containing only doublets
    """
    no_doublets_df = rt_df [~rt_df.duplicated(subset=["cc_id", "smiles.std"], keep=False)]
    doublets_df = rt_df [rt_df.duplicated(subset=["cc_id", "smiles.std"], keep=False)]
    path2file = os.path.join(path2dir, "doublets.tsv")
    doublets_df.to_csv(path2file, sep="\t", index=False)

    return no_doublets_df, doublets_df

def _treat_doublets(only_doublet_df, path2save, doublet_threshold=DOUBLET_THRESHOLD):
    """
        This functions treats the doublets. The difference is calculated as max(doublet_RT time) - min(doublet_RT time).
        If this difference > doublet_threshold, the entire doublet will be dropped, else the only entree will be kept using the
        mean RT.
        doublet_threshold is a percentage used to implement a threshold regarding the max_rt for the repo.
        Also, 2 .tsv files are saved with names "treated_doublets.tsv" and "dropped_doublets.tsv", as their name indicates, they contain inforamtion
        for the treated doublets and the dropped doublets.
        NOTE: max() and min() are used because there are "plus-que-doublets" situation, which consists of >2 identical SMILES having
        different RTs.
    """
    row_array = []
    dropped_doublets_array = []
    grouped_df = only_doublet_df.groupby (["cc_id", "smiles.std"])
    for _,doublet in grouped_df:
        diff = doublet["rt"].max() - doublet["rt"].min()
        # Should not be a problem, as the max_rt should be the same for a repo
        temp_threshold = doublet_threshold * doublet.iloc[0] ["max_rt"]
        if diff > temp_threshold:
            dropped_doublets_array.append(doublet)
        else:
            temp_row = doublet.iloc[[0]]
            temp_row ["rt"] = doublet["rt"].mean()
            row_array.append (temp_row)
    treated_doublets = pd.concat(row_array, ignore_index=True)
    dropped_doublets = pd.concat(dropped_doublets_array, ignore_index=True)

    path2file = os.path.join (path2save, "treated_doublets.tsv")
    path2droppedfile = os.path.join(path2save, "dropped_doublets.tsv")

    treated_doublets.to_csv(path2file, sep='\t', index=False)
    dropped_doublets.to_csv (path2droppedfile, sep='\t', index=False)

    return treated_doublets

def _merge_treated_doublets (no_doublets_rt_df,
                             treated_doublets_df):
    """
        Merges the no-containing doublet and the treated_doublets df together.
        Finally, a sorting will be done by "molecule_id".
    """
    unsorted_df = pd.concat([no_doublets_rt_df,treated_doublets_df],
                            ignore_index=True)
    sorted_df = unsorted_df.sort_values(by=["molecule_id"])
    return sorted_df

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
    _get_max_mean_rt_per_cc(rt_df)
    no_doublets_df, doublets_df = _find_doublets(rt_df, path2dir)
    treated_doublets = _treat_doublets(doublets_df, path2dir)
    final_df = _merge_treated_doublets(no_doublets_df, treated_doublets)

    path2file = os.path.join(path2dir, "processed_rt_data.tsv")
    final_df.to_csv(path2file, sep="\t", index=False)

    return final_df

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
    dropped_repos = grad_df.loc [~grad_df["cc_id"].isin(final_df["cc_id"]), "cc_id"].values

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
    dropped_repos = grad_df.loc [~grad_df["cc_id"].isin(final_df["cc_id"]), "cc_id"].values
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

    rt_cc_df = pd.merge (rt_df, cc_df, on = "cc_id", how = "inner")
    final_df = pd.merge (rt_cc_df, grad_df, on = "cc_id", how = "inner")

    path2file = os.path.join(path2dir, "complete_processed_data.tsv")
    final_df.to_csv(path2file, sep="\t", index=False)

    return final_df



# Main function

def get_processed_df_from_raw (source_path = SOURCE_PATH,
                               drop_smrt = True,
                               down_grad_filter = False,
                               smrt_id = "cc_127",
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
    no_same_cc_rt_df, no_same_cc_cc_df, no_same_cc_grad_df = _duplicated_cc_main(PATH2RAW_RT,
                                                                                 PATH2RAW_CC,
                                                                                 PATH2RAW_GRAD,
                                                                                 rt_df,
                                                                                 cc_df,
                                                                                 grad_df,
                                                                                 DROP_REPO,
                                                                                 source_path)
    print ("Cheking for the output directory...")
    if drop_smrt:
        downsampling = False
        no_same_cc_rt_df = no_same_cc_rt_df[no_same_cc_rt_df["cc_id"] != smrt_id]
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
    processed_rt_df = _get_processed_rt_df(no_same_cc_rt_df,
                                           path2dir=path2dir,
                                           downsampling=downsampling)

    print ("Getting processed cc data...")
    processed_cc_df  =_drop_cc_columns(no_same_cc_cc_df,
                                       path2dir)

    print ("Getting processed grad_data...")
    processed_grad_df = _get_processed_grad_df(no_same_cc_grad_df,
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
    get_processed_df_from_raw(drop_smrt=True,
                              down_grad_filter=False,)
