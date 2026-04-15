"""
    Name: RepoRT_get_raw_data.py
    Author: Yixi Zhang
    Description: This Script is based on Request to RepoRT as the URLs to all the files can easily be built with for loops.
    The seed URL is the following:
    https://raw.githubusercontent.com/michaelwitting/RepoRT/refs/heads/master/processed_data/
    This Script will get the complete raw data from RepoRT. Outputs 7 files if this file is ran:
        1. raw_rt_data.tsv (This file contains only the molecular information and the retention time)
        2. raw_column_meta.tsv (This file contains the raw column metadata with no any preprocessing).
        3. raw_gradient_data (This file contains the raw gradient data information).
        4. raw_complete_data (This file contains the raw complete data information, basically an inner join of all the 3 files above).
        5. Report_cc.txt
        6. Report_grad.txt
        7. Report_rt.txt
    The last 3 file contains the metadata of RepoRT such as missing directories, missing values, etc..
    All the NA values have been maintained, if called from another Script they must be treated.
    NOTE: Some functions have been adapted from original RepoRT_get_all_data.py (RepoRT_preprocessing.py now), so removed from there as well.

"""

# IMPORT MODULES

import os
import urllib.error

import numpy as np
import pandas as pd

# INTERNAL VARIABLES
SEED_URL = "https://raw.githubusercontent.com/michaelwitting/RepoRT/refs/heads/master/processed_data/"
PATH2RES = os.path.join(".", "data", "RepoRT", "raw_data/")
REPOS = np.arange(1, 439)

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

def _write_rt_report_file(existing_array,
                          missing_array,
                          no_isomeric_array,
                          path2res,
                          filename):
    """
        Write the Report file for the results after fetching the Retention Time data.
        3 information is written: The Repo index for existing repositories, the missing and not contained isomeric data.
    """
    report_file = os.path.join(path2res,filename)
    with open(report_file,"w") as f:
        f.write("The Repos that exist and contain files are:\n")
        f.writelines(f"{index}\n" for index in existing_array)
        f.write("The Repos that does not contain isomeric data are:\n")
        f.writelines(f"{index}\n" for index in no_isomeric_array)
        f.write("The repos that are missing are:\n")
        f.writelines(f"{index}\n" for index in missing_array)

def _write_cc_report_file(imputation_array,
                          missing_array,
                          path2res,
                          filename):
    """
        Writes 2 information:
            - Those repositories that need imputation in columns.
            - Those repositories that are missing or empty.
    """

    report_file = os.path.join(path2res,filename)
    with open(report_file,"w") as f:
        f.write("The repositories that need imputation are:\n")
        f.writelines(f"{index}:{array}\n" for dictionary in imputation_array for index, array in dictionary.items())
        f.write("The repositories that are missing or empty are:\n")
        f.writelines(f"{index}\n" for index in missing_array)

def _write_grad_report_file(not_found_array,
                            empty_array,
                            time_missing_array,
                            fr_imputation_array,
                            segment_array,
                            path2res,
                            filename):
    """
        Writes many report files:
             - Those repositories which gradient.tsv is missing or simply missing.
             - Those repositories containing gradient data without time for each segment.
             - Those repositories missing the flow rate data.
             - The segments for each repository used for building the datatable.
    """
    report_file = os.path.join(path2res, filename)
    with open(report_file,"w") as f:
        f.write("The repositories that do not contain gradient.tsv are:\n")
        f.writelines(f"{index}\n" for index in not_found_array)
        f.write("The repositories that contain empty gradient.tsv are:\n")
        f.writelines(f"{index}\n" for index in empty_array)
        f.write("The repositories that are missing the time for segments are:\n")
        f.writelines(f"{index}\n" for index in time_missing_array)
        f.write("The repos that are missing the flow rate data are:\n")
        f.writelines(f"{index}\n" for index in fr_imputation_array)
        f.write("The segments for each repository used for building the datatable are:\n")
        f.writelines(f"{index}: {array} segments\n" for dictionary in segment_array for index, array in dictionary.items())
#
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
    existing_index = []
    isomeric_not_found_array = []
    index_not_existing_array = []

    for index in index_array:
        can_url = f"{seed_url}{index}/{index}_rtdata_canonical_success.tsv"
        iso_url = f"{seed_url}{index}/{index}_rtdata_isomeric_success.tsv"
        print(f"Fetching RT data for nº{index}...")
        try:
            temp_dataframe_can = pd.read_csv(can_url, sep="\t", encoding="utf-8")
            existing_index.append(index)
            # This nested try statement avoids a not frequent but has cases where isomeric file does not exist at all, but the canonical exists.
            try:
                temp_dataframe_iso = pd.read_csv(iso_url, sep="\t", encoding="utf-8")
                temp_dataframe_iso = temp_dataframe_iso.set_index("id")
                temp_dataframe_can = temp_dataframe_can.set_index("id")
                temp_dataframe_can.update(temp_dataframe_iso)
                temp_dataframe_can = temp_dataframe_can.reset_index()
            except (urllib.error.HTTPError, pd.errors.EmptyDataError):
                print(f"Isomeric data is not found for repo nº{index}")
                isomeric_not_found_array.append(index)
            temp_df_array.append(temp_dataframe_can)
        except (urllib.error.HTTPError, pd.errors.EmptyDataError):
            index_not_existing_array.append(index)
            print(f"The repo nº {index} has not been found in the dataset. It will be skipped...")

    #Get the final array with dir_id column in the first column
    final_df = pd.concat(temp_df_array, ignore_index=True)
    dir_id_array = [str(idmol).split("_")[0] for idmol in final_df["id"]]
    final_df.insert(0, "dir_id", dir_id_array)
    final_dataframe = final_df.rename(columns={"id": "molecule_id"})

    # Write the final outputs:
    rt_data_file = os.path.join(path2res,filename)
    rt_metadata_file = "Report_rt.txt"
    final_df.to_csv(rt_data_file, sep="\t", index=False)
    _write_rt_report_file(existing_index,
                          index_not_existing_array,
                          isomeric_not_found_array,
                          path2res,
                          rt_metadata_file
                          )
    return final_dataframe

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
    not_found_or_empty_array = []
    imputation_array = []

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
                imputation_array.append({index:na_count_serie[na_count_serie != 0].index.tolist()})

        except (urllib.error.HTTPError, pd.errors.EmptyDataError):
            print (f"The repo nº {index} has not been found in the dataset or it is empty. It will be be skipped...")
            not_found_or_empty_array.append(index)

    final_df = pd.concat(cc_array, ignore_index=True)
    final_df ["id"] = existing_index_array
    final_df.rename (columns={"id": "dir_id"}, inplace=True)

    # Write the output files
    path2file = os.path.join(path2res,filename)
    report_file = "Report_cc.txt"

    final_df.to_csv(path2file, sep="\t", index=False)
    _write_cc_report_file(imputation_array,
                          not_found_or_empty_array,
                          path2res,
                          report_file)
    return final_df

def _fetch_raw_gradient_data(seed_url=SEED_URL,
                            repos_array=REPOS,
                            path2res=PATH2RES,
                            filename="raw_grad_data.tsv"):
    """
        This function builds a dataframe with only the column metadata.
        The format is slightly different from its initial form, as it is flattened into a single row.
    """
    grad_array = []
    empty_array = []
    index_array = _num2index (repos_array)
    existing_index_array = []
    not_found_array = []
    time_missing_array = []
    fr_imputation_array = []
    segments_array = []
    for index in index_array:
        print(f"Fetching gradient for {index}")
        grad_url = f'{seed_url}{index}/{index}_gradient.tsv'
        try:
            gd_df = pd.read_csv(grad_url, sep="\t", encoding="utf-8")
            if len(gd_df.index) == 0:
                empty_array.append(index)
                print (f"The gradient_data for {index} is empty.")
            elif gd_df ["t [min]"].isna().any():
                print(f"The gradient data for {index} missing the time. It will not be used.")
                time_missing_array.append(index)
            else:
                final_gd_row_df = pd.DataFrame()
                print(f"The gradient data for repo {index} will be added...")
                existing_index_array.append(index)
                if gd_df.shape [1] < 5:
                    gd_df["C [%]"] = np.zeros(gd_df.shape[0])
                    gd_df["D [%]"] = np.zeros(gd_df.shape[0])
                    gd_df = gd_df[["t [min]", "A [%]", "B [%]", "C [%]", "D [%]", "flow rate [ml/min]"]]
                segments_array.append({str(index):len(gd_df)})
                if gd_df["flow rate [ml/min]"].isna().any():
                    print(f"The repo {index} is missing the flow rate")
                    fr_imputation_array.append(index)
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
            not_found_array.append(index)
    final_df = pd.concat(grad_array, ignore_index=True)

    # Write the output files
    grad_file = os.path.join(path2res,filename)
    final_df.to_csv(grad_file, sep="\t", index=False)
    _write_grad_report_file(not_found_array,
                            empty_array,
                            time_missing_array,
                            fr_imputation_array,
                            segments_array,
                            path2res, "Report_grad.txt")
    return final_df


def merge_complete_file(path2res=PATH2RES):
    """
        This function gets the merged df with all retention time data, gradient data and cc_data
    """
    print(f"Making the output directory: {path2res}...")
    os.makedirs(path2res, exist_ok=True)

    rt_data = _fetch_raw_rt_data()
    cc_data = _fetch_raw_column_metadata()
    grad_data = _fetch_raw_gradient_data()
    rt_cc_df = pd.merge(rt_data, cc_data, on="dir_id", how="inner")
    complete_df = pd.merge(rt_cc_df, grad_data, on="dir_id", how="inner")
    path2file = os.path.join(path2res,"complete_raw_data.tsv")
    complete_df.to_csv(path2file, sep="\t", index=False)



if __name__ == "__main__":
    merge_complete_file()


