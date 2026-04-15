"""
    Name: RepoRT_preprocessing.py
    Author: Yixi Zhang
    Version: 2.0
    Description: This Script does the preprocessing of the RepoRt raw data. The preprocessing is done separately to gradient data, molecule data and gradient data.
    This processing achieves these objectives:
        1. Change retention time unit: min -> s
        2. Update formulas for molecules using the Inchi or the SMILES.
        3. Change units for the metadata.
        4. Remove all the .units columns from metadata.
        5. Implement imputation to metadata. The strategy for now is the mean value for column metadata and 0s for all the eluent data.
            5.1. The t0 value fo the columns is also inferrd from the imputed values.
        6. The column.usp.code is OneHotEncoded.
        7. Eliminates the flow rate [ml/min] from the gradient_data
    At the same time, all processed gradient data, column metadata and molecule data will be store separately as well as the complete preprocessed file.
    NO Report file will be made here as no modifications on Repo number has been done.

    Also, only the last function will be exported.
    Since this has been massively updated, the structure of the repo is changed and there are impact on other Scripts so these will also be fixed.

"""


# Import modules

import os

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.preprocessing import OneHotEncoder

from rdkit.Chem.inchi import MolFromInchi
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from src.RepoRT_data_processing.RepoRT_get_raw_data import merge_complete_file

#PARAMETERS

PATH2DIR = os.path.join (".", "data", "RepoRT", "preprocessed_data/")
PATH2INPUTS = os.path.join(".", "data", "RepoRT", "raw_data/")
RT_INPUT = os.path.join(PATH2INPUTS, "raw_rt_data.tsv")
CC_INPUT = os.path.join(PATH2INPUTS, "raw_cc_data.tsv")
GRAD_INPUT = os.path.join(PATH2INPUTS, "raw_grad_data.tsv")


# HELPER FUNCTIONS

def _get_input_df (rt_input = RT_INPUT,
                   cc_input = CC_INPUT,
                   grad_input = GRAD_INPUT):
    """
        This function checks for the input files and loads then as pd.DataFrames
    """
    print ("Cheking fot the input files...")
    if (not Path(rt_input).exists() or
        not Path(cc_input).exists() or
        not Path(grad_input).exists()):
        merge_complete_file()
    else:
        print ("The input files exists!")

    print ("Fetching the input tables...")
    rt_df = pd.read_csv(rt_input, sep='\t',dtype={"dir_id":str})
    cc_df = pd.read_csv(cc_input, sep='\t', dtype={"dir_id":str})
    grad_df = pd.read_csv(grad_input, sep='\t', dtype={"dir_id":str})

    return rt_df, cc_df, grad_df

def _get_molecule_name (column_name):
    """
        Given a column name in this format: Eluent.A.mol_name
        A string only containing the mol_name will be returned.
    """
    molecule = column_name.split(".")[2]
    return str(molecule)

def _infer_t0_val (diameter, length, fr):
    """
        Used for inferring the t0. Here t0 is calculated as V0/T.
        In RepoRT, inner diameter is given in cm, length in mm and fr in mL/min.
        So we have to pass length (mm) to cm.
    """
    base_area = np.pi * (diameter / 2)**2
    return round(((0.66*base_area*length/10)/fr)/100, 5)

# RT data preprocessing functions
def _transform_min2s(df):
    """
        This functions converts minutes to s and renames the column from "rt" to "rt_s"
    """
    df["rt"] = round(df["rt"] * 60, 2)
    final_df = df.rename({"rt":"rt_s"})
    return final_df

def _get_mol_formula_by_inchi (inchi):
    """
        This function uses RDkit to get molecule object from InChI.
    """
    mol = MolFromInchi(inchi)
    formula = CalcMolFormula(mol)
    return formula

def _get_mol_formula_by_smiles (smiles):
    """
        This function uses RDkit to get molecule object from SMILES
    """
    mol = MolFromSmiles(smiles)
    formula = CalcMolFormula(mol)
    return formula

def _get_new_formula (df):
    """
        Input: RT dataframe with a column named "formula".
        Returns the dataframe with the "formula" column updated with formulas calculated with RDkit.
        First tries with the Inchi, if can not calculate the formula, SMILES will be used.
        If none of above notation works, the original formula will be used instead.
    """
    formula_array = []
    for index, row in df.iterrows():
        try:
            formula = _get_mol_formula_by_inchi(row["inchi.std"])
            formula_array.append (formula)
        except:
            try:
                formula = _get_mol_formula_by_smiles(row["smiles.std"])
                formula_array.append (formula)
            except:
                formula_array.append (row["formula"])
    df ["formula"] = formula_array
    return df

def _obtain_preprocessed_rt_df (rt_df,
                                path2dir=PATH2DIR,
                                filename="preprocessed_rt_data.tsv"):
    """
        Preprocesses the RT data and saves it as a tsv file
    """
    print ("Converting the min to seconds...")
    rt_df = _transform_min2s(rt_df)

    print("Updating the formulas...")
    final_df = _get_new_formula(rt_df)
    final_df["dir_id"] = [str(idmol).split("_")[0] for idmol in final_df["id"]]

    path2file = os.path.join(path2dir, filename)
    print(f"Saving the preprocessed RT in {path2file}")
    final_df.to_csv(path2file, sep='\t', index=False)

    return final_df

# CC DATA PREPROCESSING

def _process_column_data (df):
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
            temp_t0 = _infer_t0_val(np.float64(row["column.id"]),
                                   np.float64(row["column.length"]),
                                   np.float64(row["column.flowrate"]))
            df.loc[index, "column.t0"] = temp_t0
        else:
            continue
    return df #This df contains the updated column metadata

def _process_eluent_unit (df):
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
                mol_name = _get_molecule_name (mol_column)
                scale_factor = mws[mol_name] / 10000 # Mw/10000
                new_value = row[df.columns[col_index -2]] * scale_factor #mM*Mw/10000
                df [mol_column] =  df [mol_column].astype(np.float64) #Necessary because the dtype in the original dset is np.int64
                df.loc[index, mol_column] = np.float64(new_value)
            elif row [column] == "µM": # If the unit is "uM", we convert the value to %(m/v)
                mol_column = df.columns[col_index - 2]
                mol_name = _get_molecule_name(mol_column)
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

def _get_one_hot_encoded_df (df):
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

def _obtain_preprocessed_cc_data(cc_df,
                                 path2dir = PATH2DIR,
                                 filename="preprocessed_cc_data.tsv"):
    """
        This function preprocesses the cc data and save it the output directory.
        This fills all the NaN values found in the eluent data part
    """

    print("Preprocessing cc data...")

    cc_df = _process_column_data(cc_df)
    cc_df= _process_eluent_unit(cc_df)
    cc_df = _get_one_hot_encoded_df(cc_df)
    filled_df = cc_df.loc [:, "eluent.A.h2o":].fillna(0)
    cc_df.loc[:, "eluent.A.h2o":] = filled_df.values

    path2file = os.path.join(path2dir, filename)

    print(f"Exporting the preprocessed cc data as {path2file}...")
    cc_df.to_csv(path2file, sep='\t', index=False)

    return cc_df


# TREATING GRADIENT INFORMATION


def _drop_flow_rate (grad_df,
                     path2dir = PATH2DIR,
                     filename="preprocessed_gradient_data.tsv"):
    """
        Drops all the columns that contain flow rate data. As this data repeats in every segment, it should be redundant.
    """
    print("Preprocessing the gradient data...")
    drop_column_array = []
    for column in grad_df.columns:
        if "flow rate" in column:
            drop_column_array.append (column)
        else:
            continue
    grad_df = grad_df.drop(drop_column_array, axis =1)

    path2file = os.path.join(path2dir, filename)
    print(f"Saving the preprocessed gradient data as {path2file}...")
    grad_df.to_csv(path2file, sep='\t', index=False)

    return grad_df

def get_preprocessed_datatable(path2dir = PATH2DIR):
    """
        This is the main function to export. As it will build the preprocessed datatable as well as writing each one of the preprocessed rt, cc and gradient data.
        This also includes getting raw data if not exists.
    """
    # Make the dir if not existing
    os.makedirs (path2dir,  exist_ok =True)

    # Input checking
    rt_df, cc_df, grad_df = _get_input_df()

    # RT preprocessing
    preprocessed_rt_df = _obtain_preprocessed_rt_df(rt_df)
    preprocessed_cc_df = _obtain_preprocessed_cc_data(cc_df)
    preprocessed_grad_df = _drop_flow_rate(grad_df)

    print ("Making the complete preprocessed datatable...")
    rt_cc_df = pd.merge(preprocessed_rt_df, preprocessed_cc_df, on="dir_id", how="inner")
    final_df = pd.merge(rt_cc_df, preprocessed_grad_df, on="dir_id", how="inner")
    final_df["dir_id"] = [str(idmol).split("_")[0] for idmol in final_df["id"]]
    path2complete_file = os.path.join(path2dir, "complete_preprocessed_data.tsv")

    print(f"Saving the complete preprocessed data as {path2complete_file}...")
    final_df.to_csv(path2complete_file, sep='\t', index=False)


#Get raw datatable
if __name__ == "__main__":
    get_preprocessed_datatable()


