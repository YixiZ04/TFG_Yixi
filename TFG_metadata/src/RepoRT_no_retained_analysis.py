"""
    This Script get the merged retained and non-reteined molecules from RepoRT:
    NOTE: processed_data ONLY. Meaning that the dataset used:
        1. Does not contain SMRT dataset (0186)
        2. Only repos with >= 100 molecules.
        3. NO doublets. In this run, the threshold is 5% of max_RT.
"""


from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import numpy as np

from src.RepoRT_data_processing.void_time_detection import detect_putative_void_time, plot_void_detection


# Functions

def get_annotation_and_plot (df,
                             rt_col,
                             units):
    """
        This is a adaptation of the main function from example.py.
        The objective of this function is to get an annotated dataframe for each subset repo.
    """

    annotated_df, result = detect_putative_void_time(
        df,
        rt_col=rt_col,
        units=units
    )

    return annotated_df, result

def save_fig_per_repo (annotated_df,
                       result,
                       path2res,
                       filename) -> None:
    """
        Get the plot for the KDE, with the annotated df and the results.
    """
    path2fig = os.path.join(path2res,filename)

    fig = plot_void_detection(annotated_df ["rt_min"].to_numpy(),
                              result,)
    fig.savefig(path2fig, dpi=300, bbox_inches='tight')
    matplotlib.pyplot.close(fig)


def main () -> None:
    input_path = os.path.join (".", "data", "RepoRT_RP", "processed_data", "with_SMRT", "processed_rt_data.tsv")
    dropped_path = os.path.join (".", "data", "RepoRT_RP", "processed_data", "with_SMRT", "report_files",
                                 "dropped_for_non_retained.tsv")

    only_retained_df = pd.read_csv(input_path, sep="\t")
    non_retained_df = pd.read_csv (dropped_path, sep="\t")
    df = pd.concat([only_retained_df, non_retained_df], ignore_index=True)
    print ("Making the saving directory...")
    path2fig = os.path.join (os.getcwd(), "TFG_metadata","AppendixA", "results_plots/")


    os.makedirs(path2fig, exist_ok=True)

    cc_id_array = df["cc_id"].unique()

    # temp_df_array = []
    print ("Iteration through data...")
    for cc_id in cc_id_array:
        temp_filename = f"{cc_id}_retention_plot.png"
        temp_df = df[df["cc_id"] == cc_id]
        temp_annotated_df, temp_result = get_annotation_and_plot(temp_df,
                                                                 rt_col = "rt",
                                                                 units = "seconds")
        save_fig_per_repo(temp_annotated_df,
                          temp_result,
                          path2fig,
                          temp_filename)
    print ("Saving the datatable")

    print (f"All the results are saved in {path2fig}")


if __name__ == "__main__":
    main()


