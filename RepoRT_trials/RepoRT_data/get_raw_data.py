"""
Here the retention time data is fetched from RepoRT and a .tsv file is build from it. Not only t is more suitable than fetching it as
a request every time we want to train models (MPNN) with it, but also, it would be easier if we wanted to get mol descriptors or column conditions.
"""


# 0. Import modules

import pandas as pd

# 1.Define some input to the function.
seed_url = 'https://raw.githubusercontent.com/michaelwitting/RepoRT/refs/heads/master/processed_data/'
save_dir = "C:/Users/leonz/Github_reps/TFG_git/RepoRT_trials/RepoRT_data/RepoRT_data.tsv"

# 2. Define get data function (canonical + isomeric SMILES and the retention time)
def get_raw_data(num_repos, save_dir=save_dir, seed_url=seed_url):
    """
    Input: num_repos to fetch, and the seed url for RepoRT.
    Output: a table containing the data fetched from RepoRT
    """
    final_dataframe = pd.DataFrame()
    patience = 15
    failed_attemps = 0
    while failed_attemps < patience:
        for num in range (num_repos):
            seed_num = str (num)
            while len (seed_num) < 4:
                seed_num = "0" + seed_num
            can_url = seed_url + seed_num + "/" + seed_num + "_rtdata_canonical_success.tsv"
            iso_url = seed_url + seed_num + "/" + seed_num + "_rtdata_isomeric_success.tsv"
            print (f"Fetching repo nº{seed_num}...")
            try:
                temp_dataframe_can = pd.read_csv(can_url, sep="\t")
                temp_dataframe_iso = pd.read_csv(iso_url, sep="\t")
                final_dataframe = pd.concat ([final_dataframe, temp_dataframe_can, temp_dataframe_iso], ignore_index=True)
                print (f"Data successfully fetched from {seed_num}")
            except:
                failed_attemps += 1
                print (f"The repo {seed_num} does not exist ")
    del temp_dataframe_can, temp_dataframe_iso
    final_dataframe.to_csv(save_dir, sep="\t", index=False)

get_raw_data (450)

# 2. Get clean table
# Load the table here as a dataframe
data_file = "C:/Users/leonz/Github_reps/TFG_git/RepoRT_trials/RepoRT_data/RepoRT_data.tsv"
report_df = pd.read_csv(data_file, sep="\t")

#Define the columns of our interest
data = {"id":report_df["id"],
        "formula":report_df["formula"],
        "smiles":report_df["smiles.std"],
        "inchi":report_df["inchi.std"],
        "rt":report_df["rt"],
        "rt_s":round(report_df["rt"]*60,2),}
clean_dataframe = pd.DataFrame(data).dropna () #In case of any NA value
clean_filename = "C:/Users/leonz/Github_reps/TFG_git/RepoRT_trials/RepoRT_data/RepoRT_data_clean.tsv"
clean_dataframe.to_csv(clean_filename, sep="\t", index=False)

