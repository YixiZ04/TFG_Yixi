import pandas as pd


base_url = "https://raw.githubusercontent.com/michaelwitting/RepoRT/refs/heads/master/processed_data/"
def fetch_grad_data (index, base_url=base_url):
    url = f"{base_url}{index}/{index}_gradient.tsv"
    temp_df = pd.read_csv (url, sep = '\t')
    return temp_df

gradient_data = fetch_grad_data ("0001")
gradient_data = gradient_data.set_index("file")
gradient_rearrangement = []
for position in range(gradient_data.shape[0]):
    gra_row = gradient_data.iloc[position, :]
    sort_gra_row = gra_row.iloc[1:5].sort_values(ascending=False)
    drop_gra_columns = sort_gra_row[2:].index
    gradient = gra_row.drop(drop_gra_columns)
    concat_column_data = pd.concat([eluent_data.loc[gradient_data.index[0]], gradient])
    drop_elu_columns = [i for i in concat_column_data.index if
                        drop_gra_columns[0][0] in i or drop_gra_columns[1][0] in i]
    eluent_gradient_data = concat_column_data.drop(drop_elu_columns)
    for col in eluent_gradient_data.index:
        if f'{gradient.index[1][0]}' in col:
            eluent_gradient_data = eluent_gradient_data.rename(index={col: f'eluent.1{col[8:]} {position}'})
        elif f'{gradient.index[2][0]}' in col:
            eluent_gradient_data = eluent_gradient_data.rename(index={col: f'eluent.2{col[8:]} {position}'})
    eluent_gradient_data = eluent_gradient_data.rename(
        index={f't [min]': f't {position}', f'flow rate [ml/min]': f'flow_rate {position}'})
    gradient_rearrangement.append(eluent_gradient_data)