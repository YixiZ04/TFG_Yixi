"""
Name: RepoRT_get_all_data.py
Author: Yixi Zhang
Date: March 2026.
Version: 1.1.
This will get the raw data from RepoRT and merge them into one single dataframe them export it to a tsv file.
Some preprocessing to the data will be done:
    1. All NA values would be filled with zero in the gradient data, but the flowrate values.
    2. All the columns that have not gradient data included will be dropped.
    3. New column "id" will be added, this refers to the dataset's id, e.g., "0001, "0002", "0003"...
    4. The "formula" column will be updated with the real formula for the molecule usign RDkit.
    5. Add a column "rt_s" containing the retention time converted to seconds.
    6. A pre-filter fot those repositories that gradient time data is not available.
This Script would need a version update, only if the format of RepoRT were changed, namely: (I lied)
the mol_data's url no longer being:https://raw.githubusercontent.com/michaelwitting/RepoRT/refs/heads/master/processed_data/{repo_index}/{repo_index}_rtdata_canonical_success.tsv
the gradient_data's url no longer being: https://raw.githubusercontent.com/michaelwitting/RepoRT/refs/heads/master/processed_data/{repo_index}/{repo_index}_gradient.tsv
or
the metadata's url no longer being: https://raw.githubusercontent.com/michaelwitting/RepoRT/refs/heads/master/processed_data/{repo_index}/{repo_index}_metadata.tsv
or any format inside those files were changed,
If any repo were add lately in RepoRT, only need to re-run the file changing the variable "num_repos".

NOTE: Internet connexion is required and takes several minutes to build thw datafile from scratch

Update: In the previous version, the doublets was not considered, meaning that if the isomeric SMILES exists for a repo, the canonical and the isomeric SMILES were both included in the dataset;
so now in this new version, this is fixed by overwriting the canonical SMILES with the isomeric SMILES if possible.
Also, the flowrate data has also been better treated, as in the previous version, the nan values in flow_rate data was not considered, so in the final datafile it is 0 in those repositories
where flowrate = nan; in this version, this has been fixed by filling them with mean flowrate values.
"""
# Import modules

import numpy as np
import pandas as pd
import os
from rdkit.Chem.inchi import MolFromInchi
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from sklearn.preprocessing import OneHotEncoder
import sys

# Define the base url for fetching metadata and gradient data from RepoRT. This process will be ignored if already done once.
seed_url = "https://raw.githubusercontent.com/michaelwitting/RepoRT/refs/heads/master/processed_data/"

#The saving dir for the final datatable
save_dir = "./data/no_extra_mol_desc/"
num_repos = 440                                   # As at this moment, only exists repo to 0438.


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

def infer_t0_val (diameter, length, fr):
    """
    Used for inferring the t0. Here t0 is calculated as V0/T.
    In RepoRT, inner diameter is given in cm, length in mm and fr in mL/min.
    So we have to pass length (mm) to cm.
    """
    base_area = np.pi * (diameter / 2)**2
    return round(((0.66*base_area*length/10)/fr)/100, 5)

def get_mol_formula_by_inchi (inchi):
    mol = MolFromInchi(inchi)
    formula = CalcMolFormula(mol)
    return formula

def get_mol_formula_by_smiles (smiles):
    mol = MolFromSmiles(smiles)
    formula = CalcMolFormula(mol)
    return formula
#GETTING RETENTION TIME DATA

def get_rt_data (num_repos=num_repos,seed_url=seed_url):
    """
    Input: num_repos to fetch, and the seed url for RepoRT.
    Output: a table containing the RT data fetched from RepoRT.
    Update: fixed the concept error, as now the isomeric success will only be used for updating the cononical success data table.
    """
    final_dataframe = pd.DataFrame()
    index_array = get_index_array(num_repos)
    for index in index_array:
        can_url = f"{seed_url}{index}/{index}_rtdata_canonical_success.tsv"
        iso_url = f"{seed_url}{index}/{index}_rtdata_isomeric_success.tsv"
        print (f"Fetching RT data for nº{index}...")
        try:
            temp_dataframe_can = pd.read_csv(can_url, sep="\t", encoding="utf-8")
            temp_dataframe_iso = pd.read_csv(iso_url, sep="\t", encoding = "utf-8")
            # Use the id of the molecule as index followed by the updating.
            temp_dataframe_can = temp_dataframe_can.set_index ("id")
            temp_dataframe_iso = temp_dataframe_iso.set_index ("id")
            temp_dataframe_can.update(temp_dataframe_iso) #Here, if isomiric success is not emply
            temp_dataframe_can = temp_dataframe_can.reset_index ()
            final_dataframe = pd.concat ([final_dataframe, temp_dataframe_can], ignore_index=True)
        except: #The only error here is that the URL has not been found
            print (f"The repo nº {index} has not been found in the dataset. It will be skipped...")
    dir_id_array = [idmol.split("_")[0] for idmol in final_dataframe["id"]]
    dir_id_array = np.array(dir_id_array)
    final_dataframe = final_dataframe.rename(columns={"id":"molecule_id"})
    final_dataframe.insert(0, "dir_id", dir_id_array)

    # Add rt_s column
    rt_s_array = final_dataframe["rt"] * 60
    position = final_dataframe.columns.get_loc("rt") + 1
    final_dataframe.insert(position, "rt_s", rt_s_array)
    final_dataframe ["rt_s"] = round(final_dataframe ["rt_s"],2)
    del temp_dataframe_can, temp_dataframe_iso, dir_id_array
    return final_dataframe

def get_new_formula (df):
    """
    Input: RT dataframe with a column named "formula".
    Returns the dataframe with the "formula" column updated with formulas calculated with RDkit.
    First tries with the Inchi, if can not calculate the formula, SMILES will be used.
    If none of above notation works, the original formula will be used instead.
    """
    formula_array = []
    for index, row in df.iterrows():
        try:
            formula = get_mol_formula_by_inchi(row["inchi.std"])
            formula_array.append (formula)
        except:
            try:
                formula = get_mol_formula_by_smiles(row["smiles.std"])
                formula_array.append (formula)
            except:
                formula_array.append (row["formula"])
    df ["formula"] = formula_array
    return df

# TREATING METADATA AND ELUENT INFORMATION

def get_column_metadata (num_repos=num_repos, seed_url = seed_url):
    """
    Inputs: The num_repos to fetch and the seed_url. The defalt values are defined.
    Output: A df containing column metadata for each dataset.
    """
    index_array = get_index_array (num_repos)
    final_df = pd.DataFrame()
    full_index_array = []
    for index in index_array:
        filename = f"{seed_url}{index}/{index}_metadata.tsv"
        try:
            temp_df = pd.read_csv(filename, sep="\t", encoding= "utf-8") #Directly read the tsv from the url
            if temp_df.shape [0] != 0:
                print(f"Fetching column metadata for nº{index}...")
                final_df = pd.concat ([final_df, temp_df], ignore_index=True)
                full_index_array.append(index)
            else:
                print (f"The repo {index} does not contain any data. It will be skipped.")
                continue
        except:
            print (f"The repo nº {index} has not been found in the dataset. It will be be skipped...")
    full_index_array = np.array(full_index_array)
    final_df ["dir_id"] = full_index_array
    del temp_df, full_index_array
    return final_df

def process_column_data (df):
    """
    Processes a raw RepoRT metadata tsv file.
    The processing consists in:
        1. Check for NA vals in the columns from "column.name" to "column.t0".
        2. If a NA value is detected, then first check if "column.name" is NA or not:
            2.1. If so, the NA val will be filled with the global mean of that parameter.
            2.2. If not, the NA val will be filled with the mean of the same column.
        3. If the mean of the same column were to be NA as well, the global mean will be used again.
        4. With all the NA vals of the metadata filled, the t0 for those columns will be inferred:
                            t0 = V0 / F = 0.66*Vcolumn / Flow_rate
    """
    #Get a smaller df for faster iteration. The id column is not used.
    temp_df = df.loc [:, "column.name":"column.t0"]
    # Create a dictionary with the column names as keys and the GLOBAL MEANS as the values.
    means_dict = {column : round(np.mean(temp_df[column]), 2) for column in temp_df.columns [2:]}

    #The updating process
    for index,row in temp_df.iterrows():
        for column in temp_df.columns [2:]:
            if pd.isnull(row[column]) and pd.isnull(row["column.name"]):
                # If the NAME AND THE COLUMN value BOTH MISSING.
                temp_df.loc[index,column] = means_dict[column] #Global mean used
            elif pd.isnull(row[column]) and pd.notnull(row["column.name"]):
                # If the name is not missing
                # Get the mean of subset of df where the name is the same
                column_name = row["column.name"]
                temp_mean = round(temp_df[temp_df["column.name"] == column_name] [column].mean(),2)
                # Check if the mean is null. If so, global mean is used instead.
                if pd.isnull(temp_mean):
                    temp_df.loc[index,column] = means_dict[column]
                else:
                    temp_df.loc[index,column] = temp_mean
    # Update the df
    df.update (temp_df)
    #Updating t0 value
    for index, row in df.iterrows():
        if row["column.t0"] == 0:
            temp_t0 = infer_t0_val(np.float64(row["column.id"]),
                                   np.float64(row["column.length"]),
                                   np.float64(row["column.flowrate"]))
            df.loc[index, "column.t0"] = temp_t0
        else:
            continue
    del temp_df, temp_mean, temp_t0
    return df #This df contains the updated column metadata

def process_eluent_unit (df):
    """
    Input: Requires a metadata df as input (from RepoRT). This should have all the metadata from every dataset concatenated in a single df.ç
    Output: All the unit in mM or uM converted to %(m/v) and the columns containig the ".unit" information will be dropped.
    """
    # This dictionary contains the approx. molecular weight of the molecules whose unit was expressed in "mM" or "uM"
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
    # The iteration is over the rows.
    for index, row in df.iterrows():
        col_index = 0
        for column in df.columns:
            col_index += 1
            if row [column] == "mM": # If the unit is "mM", we convert the value to %(m/v)
                mol_column = df.columns[col_index - 2] #Get access to the molecule's column.
                mol_name = get_molecule_name (mol_column)
                scale_factor = mws[mol_name] / 10000 # Mw/10000
                new_value = row[df.columns[col_index -2]] * scale_factor #mM*Mw/10000
                df [mol_column] =  df [mol_column].astype(np.float64) #Necessary because the dtype in the original dset is np.int64
                df.loc[index, mol_column] = np.float64(new_value)
            elif row [column] == "µM": # If the unit is "uM", we convert the value to %(m/v)
                mol_column = df.columns[col_index - 2]
                mol_name = get_molecule_name(mol_column)
                scale_factor = mws[mol_name] / 10000000 #The only difference here.
                new_value = row[df.columns[col_index - 2]] * scale_factor #uM*Mw/10000000
                df [mol_column] =  df [mol_column].astype(np.float64)
                df.loc[index, mol_column] = new_value
            else:
                continue
    # As all concentration data is expressed in % (m/v), the unit's columns are no longer needed, so just drop them.
    # Also the columns containing any gradient information will be dropped as we will treat them in a better way.
    drop_column_array = []
    for column in df.columns:
        if ".unit" in column or "gradient." in column:
            drop_column_array.append (column)
        else:
            continue
    df = df.drop (drop_column_array, axis =1)
    del drop_column_array
    return df #This df contains all the column metadata and eluent composition data.

def get_one_hot_encoded_df (df):
    """
    Input: The metadata of RepoRT (processed previously or not) containing the column "column.usp.code".
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
    del one_hot_df
    return updated_df


# TREATING GRADIENT INFORMATION

def get_gradient_data (num_repos = num_repos, seed_url = seed_url):
    """
    Works in the same way as the previous get_column_metadata functions.
    The resulting df of each row has the gradient data for its dataset
    """
    index_array = get_index_array (num_repos)
    final_df = pd.DataFrame()
    for index in index_array:
        grad_url = f'{seed_url}{index}/{index}_gradient.tsv'
        print (f"Fetching gradient data for nº{index}...")
        try:
            temp_df = pd.read_csv(grad_url, sep="\t", encoding= "utf-8") #Directly read the tsv from the url
            if temp_df ["t [min]"].isna ().any() or len(temp_df) == 0: #There are 2 conditions because there are 2 possibilities: It could a df having empty rows or with no any rows.
                print (f"The gradient data for repo {index} is not available. This repo will be skipped...")
            else:
                temp_df_row_final = pd.DataFrame ()
                # Since nº0392, the gradient.tsv format has been changed, thus this format
                # This condition checks for that new format and change it to the original format.
                if temp_df.shape [1] < 5:
                    temp_df["C [%]"] = np.zeros(temp_df.shape[0])
                    temp_df["D [%]"] = np.zeros(temp_df.shape[0])
                    temp_df = temp_df[["t [min]", "A [%]", "B [%]", "C [%]", "D [%]", "flow rate [ml/min]"]]
                # Iteration over rows of the gradient Dataframe
                for grad_index, row in temp_df.iterrows():
                    temp_dict = {}
                    for column in temp_df.columns:
                        temp_dict [f"{column}_{grad_index}"] =row[column]
                    temp_df_row = pd.DataFrame([temp_dict])
                    temp_df_row_final = pd.concat ([temp_df_row_final, temp_df_row], axis = 1)  #This df flattens the gradient data into a single row.
                temp_df_row_final.insert (0, "dir_id", index) #Here the exp_id is added
                final_df = pd.concat ([final_df, temp_df_row_final], ignore_index = True) # Contains all the flattened gradient data for all dataset.
        except:
            print (f"The repo nº {index} has not been found in the dataset. And it will be skipped...")
    final_df = final_df.fillna (0)
    del temp_df, temp_df_row_final, temp_df_row
    return final_df

def update_flow_rate (df, max_grad_num = 18):
    """
    This function updates all flowrate in the gradient data. There are nan values in flowrates and in the fetching process, they were filled with 0.
    So here, we update those values of 0 with the flowrate inferred using the mean flowrate of all the repositories.
    This updating process is only applied to the segments of gradient where it exists (time of the segment !=0), meaning for example, if in a repo only 7 segments,
    all the other segments from 8 to final will not be filled with flowrate.
    """
    updated_df = []
    index_array = np.unique(df["dir_id"])
    for index in index_array:
        print (index)
        temp_df = df[df["dir_id"] == index].reset_index(drop = True)
        fr_array = temp_df.loc [:,"column.flowrate"].values
        for grad_index in range(max_grad_num):
            time_column_name = "t [min]_" + str(grad_index)
            fr_column_name = "flow rate [ml/min]_" + str(grad_index)
            time = temp_df.loc[0, time_column_name]
            if time != 0 and grad_index > 0:
                temp_df [fr_column_name] = fr_array
            else:
                continue
        updated_df.append (temp_df)
    return pd.concat (updated_df, ignore_index = True)

def get_raw_datatable(save_dir = save_dir):
    """
    This function makes the raw datatable from RepoRT with all data processed and saves it in the save_dir.
    If in the future there were updates in RepoRT Repos, just change the file name and re-run this script.
    """
    # Make the dir if not existing
    os.makedirs (save_dir,  exist_ok =True)
    save_file = save_dir + "RepoRT_complete_data.tsv"

    #Get rt data
    print (f"Fetching RT data from RepoRT...")
    temp_rt_data = get_rt_data()
    print (f"RT data successfully fetched !")
    # Updating the formula
    print (f"Updating the formulas for molecules...")
    temp_rt_data = get_new_formula(temp_rt_data)
    print (f"RT data successfully updated !")

    #Process the column metadata
    print (f"Fetching the column metadata...")
    temp_column_metadata = get_column_metadata ()
    print (f"The columns' metadata successfully fetched !")
    print (f"Processing the column metadata...")
    temp_column_metadata = process_column_data (temp_column_metadata)
    temp_column_metadata = process_eluent_unit (temp_column_metadata)
    temp_column_metadata = get_one_hot_encoded_df (temp_column_metadata)
    print (f"The columns' metadata successfully processed !")

    #Process the gradient data
    print (f"Fetching the gradient data...")
    temp_grad_data = get_gradient_data ()
    print (f"The gradient data successfully fetched !")

    #Inner join on "dir_id" column both to get the final datatable and save it in the save_file given
    print (f"Building the raw datatable and saving it in {save_file}...")
    temp_final_datatable = pd.merge (temp_column_metadata, temp_grad_data, on = "dir_id", how = "inner")
    final_datatable = pd.merge (temp_rt_data, temp_final_datatable, on = "dir_id", how = "inner")
    final_datatable = update_flow_rate (final_datatable)

    # Export the final datatable
    final_datatable.to_csv (save_file, sep = "\t", index = False)
    print (f"The raw datatable successfully saved !")
    del temp_column_metadata, temp_grad_data, temp_rt_data, temp_final_datatable


#Get raw datatable
if __name__ == "__main__":
    get_raw_datatable()



