"""
    Name: RepoRT_get_raw_data.py
    Author: Yixi Zhang
    Description: This Script is based on Request to RepoRT as the URLs to all the files can easily be built with for loops.
    The seed URL is the following:
    https://raw.githubusercontent.com/michaelwitting/RepoRT/refs/heads/master/processed_data/
    This Script will get the complete raw data from RepoRT. Outputs 6 files if this file is ran:
        1. raw_rt_data.tsv (This file contains only the molecular information and the retention time)
        2. raw_column_meta.tsv (This file contains the raw column metadata with no any preprocessing).
        3. raw_gradient_data (This file contains the raw gradient data information).
        4. raw_complete_data (This file contains the raw complete data information, basically an inner join of all the 3 files above).
        5. Summary.tsv -> This contains all information of the raw data: n molecules, method type, if gradient data available, segments, if needed inputation.
        6. Dropped_for_no_gradient.tsv -> Includes those repor whose gradient data is missing or the repo is non-existing.
    The last 3 file contains the metadata of RepoRT such as missing directories, missing values, etc..
    All the NA values have been maintained, if called from another Script they must be treated.
    NOTE: Some functions have been adapted from original RepoRT_get_all_data.py (RepoRT_preprocessing.py now), so removed from there as well.
    Update: Added option for fetching Reverse Phase data, to separate it from the HILIC and "other". These last 2 types might be evaluated later.

"""

# IMPORT MODULES

import os
import urllib.error

import numpy as np
import pandas as pd

# INTERNAL VARIABLES
SEED_URL = "https://raw.githubusercontent.com/michaelwitting/RepoRT/refs/heads/master/processed_data/"
PATH2RES = os.path.join(".", "data", "RepoRT", "raw_data/")
REPOS = np.arange(1, 393)

# HELPER FUNCTIONS

def _num2index (repos_array):
    """
        This function convert an array containing number to RepoRT index:
        For example: 1 -> 0001
    """
    index_array=[]

    for repo in repos_array:
        index = str(repo)
        while len(index) < 4:
            index = "0" + index
        index_array.append (index)
    return np.array(index_array)

# FUNCTIONS TO FETCH DATA
def _fetch_raw_rt_data (seed_url=SEED_URL,
                       repos_array=REPOS,
                       path2res=PATH2RES,
                       filename="raw_rt_data.tsv"):
    """
        This function fetches the rt_time and the molecular information with the url and the array given.
        It also outputs a report in the same directory for those missing directories
    """

    temp_df_array = []
    index_array = _num2index (repos_array)
    summary_dict = {}

    for index in index_array:
        can_url = f"{seed_url}{index}/{index}_rtdata_canonical_success.tsv"
        iso_url = f"{seed_url}{index}/{index}_rtdata_isomeric_success.tsv"
        print(f"Fetching RT data for nº{index}...")
        try:
            temp_dataframe_can = pd.read_csv(can_url, sep="\t", encoding="utf-8")
            # This nested try statement avoids a not frequent but has cases where isomeric file does not exist at all, but the canonical exists.
            try:
                temp_dataframe_iso = pd.read_csv(iso_url, sep="\t", encoding="utf-8")
                temp_dataframe_iso = temp_dataframe_iso.set_index("id")
                temp_dataframe_can = temp_dataframe_can.set_index("id")
                temp_dataframe_can.update(temp_dataframe_iso)
                temp_dataframe_can = temp_dataframe_can.reset_index()
                summary_dict [index] = "Both canonical and isomeric data exist"
            except (urllib.error.HTTPError, pd.errors.EmptyDataError):
                print(f"Isomeric data is not found for repo nº{index}")
                summary_dict [index] = "Only canonical data exists. Isomeric not found"
            temp_df_array.append(temp_dataframe_can)
        except (urllib.error.HTTPError, pd.errors.EmptyDataError):
            summary_dict [index] = "rt data not found"
            print(f"The repo nº {index} has not been found in the dataset. It will be skipped...")

    #Get the final array with dir_id column in the first column
    final_df = pd.concat(temp_df_array, ignore_index=True)
    dir_id_array = [str(idmol).split("_")[0] for idmol in final_df["id"]]
    final_df.insert(0, "dir_id", dir_id_array)
    final_dataframe = final_df.rename(columns={"id": "molecule_id"})

    # Write the final outputs:
    rt_data_file = os.path.join(path2res,filename)
    final_dataframe.to_csv(rt_data_file, sep="\t", index=False)
    rt_metadatadf = pd.DataFrame({
        "dir_id": summary_dict.keys(),
        "rt data status": summary_dict.values(),
        "n molecules": [ len(final_dataframe[final_dataframe["dir_id"]==index]) for index in index_array]
    })
    return final_dataframe, rt_metadatadf

def _fetch_raw_column_metadata(seed_url=SEED_URL,
                              repos_array=REPOS,
                              path2res=PATH2RES,
                              filename="raw_cc_data.tsv"):
    """
        This function builds a dataframe with only the column metadata.
        Also indicates those repositories who need imputation
    """
    cc_array = []
    index_array = _num2index (repos_array)
    existing_index_array = []

    summary_dict = {}

    for index in index_array:
        cc_url = f"{seed_url}{index}/{index}_metadata.tsv"
        try:
            print(f"Fetching column metadata for nº{index}...")
            cc_df = pd.read_csv(cc_url, sep="\t", encoding="utf-8")
            cc_array.append(cc_df)
            existing_index_array.append(index)
            temp_df = cc_df[[ col for col in cc_df.columns if ".unit" not in col]]
            na_count_serie = temp_df.isna().sum()
            if na_count_serie.sum() != 0:
                temp_list = na_count_serie[na_count_serie != 0].index.tolist()
                summary_dict [index] = "\n".join(temp_list)
            else:
                summary_dict[index] = "No need for imputation"

        except (urllib.error.HTTPError, pd.errors.EmptyDataError):
            print (f"The repo nº {index} has not been found in the dataset or it is empty. It will be be skipped...")
            summary_dict[index] = "Not found"

    final_df = pd.concat(cc_array, ignore_index=True)
    final_df ["id"] = existing_index_array
    final_df.rename (columns={"id": "dir_id"}, inplace=True)

    # Write the output files
    path2file = os.path.join(path2res,filename)
    final_df.to_csv(path2file, sep="\t", index=False)

    cc_metadata_df = pd.DataFrame({
        "dir_id": summary_dict.keys(),
        "cc_status": summary_dict.values(),
    })

    return final_df, cc_metadata_df

def _fetch_raw_gradient_data(seed_url=SEED_URL,
                            repos_array=REPOS,
                            path2res=PATH2RES,
                            filename="raw_grad_data.tsv"):
    """
        This function builds a dataframe with only the column metadata.
        The format is slightly different from its initial form, as it is flattened into a single row.
    """
    grad_array = []
    index_array = _num2index (repos_array)
    summary_dict = {
        "dir_id":[],
        "grad_status":[],
        "segment_count":[],
    }
    for index in index_array:
        print(f"Fetching gradient for {index}")
        grad_url = f'{seed_url}{index}/{index}_gradient.tsv'
        summary_dict["dir_id"].append(index)
        try:
            gd_df = pd.read_csv(grad_url, sep="\t", encoding="utf-8")
            if len(gd_df.index) == 0:
                summary_dict ["grad_status"].append("Empty data table")
                summary_dict["segment_count"].append(0)
                print (f"The gradient_data for {index} is empty.")
            elif gd_df ["t [min]"].isna().any():
                print(f"The gradient data for {index} missing the time. It will not be used.")
                summary_dict ["grad_status"].append("Empty data table")
                summary_dict["segment_count"].append(0)
            else:
                final_gd_row_df = pd.DataFrame()
                print(f"The gradient data for repo {index} will be added...")
                if gd_df.shape [1] < 5:
                    gd_df["C [%]"] = np.zeros(gd_df.shape[0])
                    gd_df["D [%]"] = np.zeros(gd_df.shape[0])
                    gd_df = gd_df[["t [min]", "A [%]", "B [%]", "C [%]", "D [%]", "flow rate [ml/min]"]]
                summary_dict["segment_count"].append(len(gd_df))
                if gd_df["flow rate [ml/min]"].isna().any():
                    print(f"The repo {index} is missing the flow rate")
                    summary_dict["grad_status"].append("Need flow rate imputation")
                else:
                    summary_dict["grad_status"].append("Gradient data ok")
                for grad_index, row in gd_df.iterrows():
                    temp_dict = {}
                    for column in gd_df.columns:
                        temp_dict [f"{column}_{grad_index}"] =row[column]
                    row_gd_df = pd.DataFrame([temp_dict])
                    final_gd_row_df = pd.concat([final_gd_row_df, row_gd_df],  axis = 1)
                final_gd_row_df.insert(0, "dir_id", index)
                grad_array.append(final_gd_row_df)
        except urllib.error.HTTPError:
            print (f"The gradient data for repo nº {index} has not been found in the dataset. And it will be skipped...")
            summary_dict["grad_status"].append("Gradient.tsv not found")
            summary_dict["segment_count"].append(0)
    final_df = pd.concat(grad_array, ignore_index=True)

    # Write the output files
    grad_file = os.path.join(path2res,filename)
    final_df.to_csv(grad_file, sep="\t", index=False)

    grad_metadata_df = pd.DataFrame(summary_dict)

    return final_df, grad_metadata_df

def _fetch_info_data(seed_url = SEED_URL,
                     repos_array=REPOS,
                     path2res = PATH2RES,
                     ):
    """
        Fetches info data from RepoRT. Returns a dictionary containing chromatography tupe as keys and an array containing corresponding indexes.
        Also writes a report file indicating the dir_id for each chromatography type.
    """
    final_dict = {}
    summary_dict = {
        "dir_id":[],
        "type":[]
    }
    index_array = _num2index(repos_array)
    for index in index_array:
        info_url = f'{seed_url}{index}/{index}_info.tsv'
        summary_dict["dir_id"].append(index)
        print (f"Fetching info data for {index}...")
        try:
            info_df = pd.read_csv(info_url, sep="\t", encoding="utf-8")
            chromatography_type = str(info_df ["method.type"] [0])
            summary_dict["type"].append(chromatography_type)
            if chromatography_type not in final_dict:
                final_dict[chromatography_type] = [index]
            else:
                final_dict[chromatography_type].append(index)
        except (urllib.error.HTTPError, pd.errors.EmptyDataError):
            print(f"The repo nº {index} has not been found in the dataset. And it will be skipped...")
            summary_dict["type"].append("Info data not found")

    return final_dict, pd.DataFrame(summary_dict)


def _get_final_summary_df (rt_summary_df,
                           cc_summary_df,
                           grad_summery_df,
                           info_summary_df):
    """
        Obtains the final summary dataframes
    """
    temp_summery_df = pd.merge(rt_summary_df, info_summary_df, on="dir_id", how="inner")
    temp_temp_summery_df = pd.merge(temp_summery_df, grad_summery_df, on="dir_id", how="inner")
    final_summary_df = pd.merge (temp_temp_summery_df, cc_summary_df, on="dir_id", how="inner")
    final_summary_df.loc["Total", "n molecules"] = final_summary_df["n molecules"].sum()

    mask = (
            (grad_summery_df["grad_status"] != "Empty data table") &
            (grad_summery_df["grad_status"] != "Gradient.tsv not fuond")
    )

    dropped_dirid = grad_summery_df.loc[mask, "dir_id"]
    dropped_df = rt_summary_df[~rt_summary_df["dir_id"].isin(dropped_dirid)]
    final_dropped_df = dropped_df[["dir_id", "n molecules"]]
    final_dropped_df.loc["Total"] = final_dropped_df.sum(numeric_only=True)

    return final_summary_df, final_dropped_df

# THE USEFUL FUNCTION
def merge_complete_file(path2res=PATH2RES):
    """
        This function gets the merged df with all retention time data, gradient data and cc_data
    """
    print(f"Making the output directory: {path2res}...")
    os.makedirs(path2res, exist_ok=True)

    rt_data, rt_summary_df = _fetch_raw_rt_data()
    cc_data,cc_summary_df = _fetch_raw_column_metadata()
    grad_data, grad_summary_df = _fetch_raw_gradient_data()
    info_data, info_summary_df = _fetch_info_data()
    rt_cc_df = pd.merge(rt_data, cc_data, on="dir_id", how="inner")
    complete_df = pd.merge(rt_cc_df, grad_data, on="dir_id", how="inner")
    path2file = os.path.join(path2res,"complete_raw_data.tsv")
    complete_df.to_csv(path2file, sep="\t", index=False)

    final_summary, final_dropped = _get_final_summary_df(rt_summary_df,
                                                         cc_summary_df,
                                                         grad_summary_df,
                                                         info_summary_df)
    path2reports = os.path.join(path2res, "report_files/")
    os.makedirs(path2reports, exist_ok=True)

    final_summary.to_csv (os.path.join(path2reports, "Summary.tsv"), sep='\t', index=True)
    final_dropped.to_csv (os.path.join(path2reports, "Dropped_for_no_gradient.tsv"), sep='\t', index=True)

    for key, index_array in info_data.items():
        new_path2dir = os.path.join (".", "data", f"RepoRT_{key}", "raw_data/")
        os.makedirs(new_path2dir, exist_ok=True)
        temp_complete_df = complete_df[complete_df["dir_id"].isin(index_array)]
        temp_rt_df = rt_data[rt_data["dir_id"].isin(index_array)]
        temp_cc_df = cc_data[cc_data["dir_id"].isin(index_array)]
        temp_grad_df = grad_data[grad_data["dir_id"].isin(index_array)]

        report_df = final_summary [final_summary["dir_id"].isin(index_array)]
        report_df.loc["Total", "n molecules"] = report_df["n molecules"].sum()
        report_df.to_csv (os.path.join(new_path2dir, f"Summary_{key}.tsv"), sep='\t', index=False)


        temp_complete_df.to_csv (os.path.join(new_path2dir, "complete_raw_data.tsv"), sep="\t", index=False)
        temp_rt_df.to_csv(os.path.join(new_path2dir, "raw_rt_data.tsv"), sep="\t", index=False)
        temp_cc_df.to_csv(os.path.join(new_path2dir, "raw_cc_data.tsv"), sep="\t", index=False)
        temp_grad_df.to_csv(os.path.join(new_path2dir, "raw_grad_data.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    merge_complete_file(path2res=PATH2RES)

