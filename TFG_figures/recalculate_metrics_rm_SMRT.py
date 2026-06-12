import os

import pandas as pd
import numpy as np

from src.training.functions.splitted_sets_functions import metrics_from_dataframe


def _get_no_smrt_folds (seed_path2folds, path2all_results):
    """
        For a better generalizartion. Individual entries will be used.
    """
    final_array = []
    all_res_df = pd.read_csv(path2all_results, sep = "\t", dtype={"cc_id":str,"molecule_id":str})
    for i in range(10):
        filename = f"fold_{i}.tsv"
        fold_df = pd.read_csv(os.path.join(seed_path2folds,filename), sep = "\t", dtype={"cc_id":str,"molecule_id":str})
        no_smrt_df = fold_df[fold_df ["cc_id"] != "cc_125"]
        unique_ids = no_smrt_df["cc_id"].unique()
        result_df = all_res_df[all_res_df["cc_id"].isin(unique_ids)]
        final_array.append(result_df)
    return final_array

def _get_metrics_df (folds_array):
    metrics_dict = {
        "MAE": [],
        "RMSE": [],
        "MRE": [],
        "rel_max_rt_error": [],
        "rel_mean_rt_error": [],
        "n molecules test": [],
    }
    for fold_df in folds_array:
        mae, rmse, mre, rel_max_error, rel_mean_error = metrics_from_dataframe(fold_df)
        # This is new
        metrics_dict["MAE"].append(mae)
        metrics_dict["RMSE"].append(rmse)
        metrics_dict["MRE"].append(mre)
        metrics_dict["rel_max_rt_error"].append(rel_max_error)
        metrics_dict["rel_mean_rt_error"].append(rel_mean_error)
        metrics_dict["n molecules test"].append(len(fold_df))

    return metrics_dict

def _get_aggregated_metrics (metric_array, size_array):
    """
        Calculates the aggregated mean based on the size of each fold.
    """
    mean_numerator = np.sum(metric_array * size_array)
    denominator = np.sum(size_array)
    agg_mean = mean_numerator / denominator

    std_numerator = np.sum (size_array * (metric_array - agg_mean)**2)
    agg_std = np.sqrt(std_numerator / denominator)
    return np.round(agg_mean, 4), np.round(agg_std, 4)

def _write_overall_results (result_dict, path2output):
    """
        This function summarizes the results from all 10 runs into a single dataframe and saved in a .tsv file.
        Also, another .tsv file is written with the mean and standard deviation for each metric.

    """
    k = len(result_dict["MAE"])
    summarized_df = pd.DataFrame (result_dict, index=([f"run_{i}" for i in range(1, k+1)]))

    temp_metric_dict = {"metric":[],
                        "aggregated mean":[],
                        "standard deviation":[]}

    size_array = summarized_df["n molecules test"].to_numpy()
    only_metric_df = summarized_df.drop(columns=["n molecules test"])

    for column in only_metric_df.columns:
        metric_array = only_metric_df[column].to_numpy()
        agg_mean, agg_std = _get_aggregated_metrics(metric_array, size_array)

        temp_metric_dict["metric"].append(column)
        temp_metric_dict["aggregated mean"].append(agg_mean)
        temp_metric_dict["standard deviation"].append(agg_std)

    metric_df = pd.DataFrame(temp_metric_dict)
    path2overall_metrics_file = os.path.join (path2output, "new_overall_metrics.tsv")
    metric_df.to_csv (path2overall_metrics_file, sep='\t', index=False)

def main():
    project_dir = os.getcwd()
    path2cc_entries = os.path.join(project_dir,
                                   "data",
                                   "RepoRT_RP",
                                   "processed_data",
                                   "with_SMRT",
                                   "kfolds",
                                   "cc_split")
    for moldesc_type in ["RepoRT_RP_kfold_moldesc", "RepoRT_RP_kfold"]:
        seed_path = os.path.join(project_dir, "results4TFG", moldesc_type, "with_SMRT", "no_filtered",
                                 "cc_split", "01_06_2026/")
        path2all_res = os.path.join(seed_path, "all_results.tsv")
        no_smrt_folds_array = _get_no_smrt_folds(path2cc_entries, path2all_res)
        metrics_dict = _get_metrics_df(no_smrt_folds_array)
        _write_overall_results(metrics_dict, seed_path)

if __name__ == "__main__":
    main()
