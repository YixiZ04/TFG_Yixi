"""
Author: Yixi Zhang
This will get the column metadata and also gradient data from RepoRT.
The main objective is to get a single df that merges the column metadata and gradient data:
    1. All NA values would be filled with zero in the gradient data.
    2. All the columns that have not gradient data included will be dreoped.
    3. A experiment id column will be added for later's inner joining.
"""


# Import modules

import numpy as np
import pandas as pd
from IPython.utils.io import temp_pyfile
from sklearn.preprocessing import OneHotEncoder

# Define the base url for fetching metadata and gradient data from RepoRT. This process will be ignored if already done once.
seed_url = "https://raw.githubusercontent.com/michaelwitting/RepoRT/refs/heads/master/processed_data/"


# Here general functions for data_treating are defined
def get_index_array (num_repos):
    """
    Input: A number of repositories wanted to fetch.
    Output: A numpy array with indexes of the repoRT data (e.g., "0001", "0002"...)
    """
    index_array = []
    for index in range(1, num_repos):
        index = str(index)
        while len (index) < 4:
            index = "0" + index
        index_array.append (index)
    return np.array(index_array)

def get_molecule_name (column_name):
    """
    Given a column name in this format: Eluent.A.mol_name
    A string only containing the mol_name will be returned.
    """
    molecule = column_name.split(".")[2]
    return str(molecule)

#Defining functions to get the metadata and treating them.

def get_column_metadata (num_repos=438, seed_url = seed_url):
    """
    Inputs: The num_repos to fetch and the seed_url. The defalt values are defined.
    Output: A df containing column metadata for each dataset.
    """
    index_array = get_index_array (num_repos)
    final_df = pd.DataFrame()
    for index in index_array:
        filename = f"{seed_url}{index}/{index}_metadata.tsv"
        try:
            temp_df = pd.read_csv(filename, sep="\t", encoding= "utf-8") #Directly read the tsv from the url
            if temp_df.shape [0] != 0:
                final_df = pd.concat ([final_df, temp_df], ignore_index=True)
                print(f"Successfully loaded repo nº {index}")
            else:
                print (f"The repo {index} does not contain any info. So it will be skipped.")
                continue
        except:
            print (f"The repo nº {index} was not found in the dataset. And it is gonna be skipped...")
    del temp_df
    return final_df

def process_metadata (df):
    """
    Input: Requires a metadata df as input (from RepoRT). This should have all the metadata from every dataset concatenated in a single df.ç
    Output: All the unit in mM or uM converted to %(m/v) and the columns containig the ".unit" information will be dropped.
    """
    # Iteration over rows in order to process the concentration info.
    mws = {
        "acetic": 60,
        "phosphor": 98,
        "nh4ac": 77,
        "nh4form": 63,
        "nh4carb": 96,
        "nh4bicarb": 79,
        "nh4f": 37,
        "nh4oh": 35,
        "trieth": 101,
        "triprop": 143,
        "tribut": 185,
        "nndimethylhex": 129,
        "medronic": 176,
    }
    for index, row in df.iterrows():
        col_index = 0
        for column in df.columns:
            col_index += 1
            if row [column] == "mM":
                mol_column = df.columns[col_index - 2] #Get access to the molecule's column.
                mol_name = get_molecule_name (mol_column)
                scale_factor = mws[mol_name] / 10000 # Mw/10000
                new_value = row[df.columns[col_index -2]] * scale_factor #mM*Mw/10000
                df [mol_column] =  df [mol_column].astype(np.float64) #Necessary because the dtype in the original dset is np.int64
                df.loc[index, mol_column] = np.float64(new_value)
            elif row [column] == "µM":
                mol_column = df.columns[col_index - 2]
                mol_name = get_molecule_name(mol_column)
                scale_factor = mws[mol_name] / 10000000 #The only difference here.
                new_value = row[df.columns[col_index - 2]] * scale_factor #uM*Mw/10000000
                df [mol_column] =  df [mol_column].astype(np.float64)
                df.loc[index, mol_column] = new_value
            else:
                continue
    # As all concentration data is expressed in % (m/v), the unit's columns are no longer needed, so just drop them.
    drop_column_array = []
    for column in df.columns:
        if ".unit" in column or ".start" in column or ".end" in column:
            drop_column_array.append (column)
        else:
            continue
    df = df.drop (drop_column_array, axis =1)
    del drop_column_array
    return df #This df contains all the column metadata and eluent composition data.

def get_one_hot_encoded_df (df):
    """
    Input: A dataframe containing repoRT data.
    Output: An updated df with new columns of USP code one-hot encoded.
    """
    encoder = OneHotEncoder()
    one_hot_data = encoder.fit_transform(df[["column.usp.code"]])
    one_hot_df = pd.DataFrame(one_hot_data.toarray(),
                              columns=encoder.get_feature_names_out(['column.usp.code']))
    position_column_name = df.columns.get_loc("column.name")
    updated_df = pd.concat([df.iloc[:, :position_column_name + 1],
                                      one_hot_df,
                                      df.iloc[:, position_column_name + 1:]], axis=1)
    return updated_df

# Get the gradient data and treat them
def get_gradient_data (num_repos = 440, seed_url = seed_url):
    """
    Works in the same way as the previous get_column_metadata functions.
    The resulting df of each row has the gradient data for its dataset
    """
    index_array = get_index_array (num_repos)
    final_df = pd.DataFrame()
    full_index_array = []
    count = 0
    for index in index_array:
        grad_url = f'{seed_url}{index}/{index}_gradient.tsv'
        try:
            temp_df = pd.read_csv(grad_url, sep="\t", encoding= "utf-8") #Directly read the tsv from the url
            if temp_df ["t [min]"].isna ().any(): #Here if any grad_data is null
                print (f"The gradient data for repo {index} is not available. This repo will be skipped...")
            else:
                temp_df_row_final = pd.DataFrame ()
                full_index_array.append (index)
                if temp_df.shape [1] < 5:
                    temp_df["C [%]"] = np.zeros(temp_df.shape[0])
                    temp_df["D [%]"] = np.zeros(temp_df.shape[0])
                    temp_df = temp_df[["t [min]", "A [%]", "B [%]", "C [%]", "D [%]", "flow rate [ml/min]"]]
                for grad_index, row in temp_df.iterrows():
                    temp_dict = {}
                    for column in temp_df.columns:
                        temp_dict [f"{column}_{grad_index}"] =row[column]
                    temp_df_row = pd.DataFrame([temp_dict])
                    temp_df_row_final = pd.concat ([temp_df_row_final, temp_df_row], axis = 1)
                temp_df_row_final.insert (0, "exp_id", index)
                final_df = pd.concat ([final_df, temp_df_row_final], ignore_index = True)
                count += 1
        except:
            print (f"The repo nº {index} was not found in the dataset. And it is gonna be skipped...")
    # final_df.insert (0, "exp_id", full_index_array)
    del temp_df
    print (str(count))
    return final_df, full_index_array


c = np.concatenate((a ["exp_id"].to_numpy, b))
np.unique (c)