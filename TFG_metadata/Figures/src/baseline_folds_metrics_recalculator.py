import os

import pandas as pd
import numpy as np

from collections import defaultdict

def _obtain_input_dict (df) -> dict:

    folds_dict = {}
    fold_count = 0
    for _,fold in df.groupby(by=["k-fold-n"]):
        folds_dict[f"fold_{fold_count}"] = fold
        fold_count += 1

    return folds_dict


def _obtain_aggregated_metric(df, metric_col, n_molecule_col) -> float:
    n_molecules = df[n_molecule_col].sum()
    aggregated_metric = np.float32(np.sum(df[metric_col]*df[n_molecule_col]/n_molecules))
    return aggregated_metric

def _obtain_aggregated_metric_df(input_dict, n_molecule_col) -> pd.DataFrame:
    metric_dict = defaultdict(list)

    for fold, df in input_dict.items():
        metric_list = df.loc[:, "MAE":]
        mols = df[n_molecule_col].sum()
        metric_dict["fold_name"].append(fold)
        metric_dict["n_molecules"].append(mols)
        for metric in metric_list:
            agg_metric = _obtain_aggregated_metric(df, metric, n_molecule_col)
            metric_dict[metric].append(agg_metric)
    return pd.DataFrame(metric_dict)

def _calculate_std (df,metric, mean_val,  n_molecule_col) -> float:
    std_numerator = np.sum (df[n_molecule_col] * (df[metric] - mean_val)**2)
    agg_std = np.sqrt(std_numerator / df[n_molecule_col].sum())
    return float(agg_std)

def _obtain_overall_agg_metrics(metric_df, n_molecule_col) -> pd.DataFrame:
    metrics = metric_df.loc[:, "MAE":]
    temp_dict = defaultdict(list)

    for metric in metrics:
        std_col = metric +"_std"
        mean = _obtain_aggregated_metric(metric_df,metric, n_molecule_col)
        std =_calculate_std(metric_df,metric, mean, n_molecule_col)
        temp_dict[metric].append(mean)
        temp_dict[std_col].append(std)
    return pd.DataFrame(temp_dict)

def main() -> None:
    seed_path = os.path.join(".", "results4TFG", "RepoRT_RP_kfold", "with_SMRT", "no_filtered",
                              "model_per_repo", "09_06_2026/")
    path2input = os.path.join(seed_path, "all_folds.tsv")

    input_df = pd.read_csv(path2input, sep="\t")

    temp_dict = _obtain_input_dict(input_df)
    metric_df = _obtain_aggregated_metric_df(temp_dict, "n_molecules")
    metric_df.to_csv(os.path.join(seed_path, "new_agg_metrics.tsv"), sep='\t', index=False)

    overall_agg_df = _obtain_overall_agg_metrics(metric_df, "n_molecules")
    overall_agg_df.to_csv(os.path.join(seed_path, "new_overall_metrics.tsv"), sep='\t', index=False)

if __name__ == "__main__":
    main()
