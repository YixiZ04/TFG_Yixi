import pandas as pd
import numpy as np

path2file1 = "/home/user/YixiTFG/final_data_nt.tsv"
path2file2 = "/home/user/YixiTFG/RepoRT_classified_CCinformation.tsv"

df_1 = pd.read_csv (path2file1, sep='\t').sort_values (by=["id"], ascending=True, ignore_index=True)
df_2 = pd.read_csv (path2file2, sep='\t', low_memory=False)

df_2 = df_2.drop (columns=["alternative_parents"])

indexes = np.array (["0002", "0003"])

def get_data_from_index (df, index):
    """
    Input: The raw dataframe containing all data and the index of the repo that you want to use for training.
    Output: A subset of the dataframe including all the molecules from the repo number.
    """
    temp_dataframe = df[df ["id"].str.split("_").str[0] == index]
    return temp_dataframe

final_df = get_data_from_index (df_2, "0001")
for index in indexes:
    temp_dataframe = get_data_from_index (df_2, index)
    final_df = pd.concat ([final_df, temp_dataframe], ignore_index=True)


columns_df1 = df_1.columns
columns_df2 = df_2.columns
diff_columns = []
count = 0
for column in columns_df1:
    if column not in columns_df2:
        count += 1
        print(column)
        diff_columns.append (column)
    else:
        continue
