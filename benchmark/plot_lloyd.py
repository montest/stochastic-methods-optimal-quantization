import os

import pandas as pd

from benchmark.benchmark_utils import plot_results_lloyd, plot_ratios_lloyd

if __name__ == "__main__":
    directory_path = "/Users/thibautmontes/GitHub/stochastic-methods-optimal-quantization/benchmark/"
    path_to_results = os.path.join(directory_path, "final_results_lloyd.csv")
    if os.path.exists(path_to_results) and os.path.getsize(path_to_results) > 0:
        df_results = pd.read_csv(path_to_results, index_col=0)
    else:
        raise Exception(f"Wrong path to benchmark results {path_to_results}")
    df_results['device'].fillna("cpu", inplace=True)
    df_results["method"] = df_results.loc[:, "method_name"] + "_" + df_results.loc[:, "device"]
    df_results["method"].replace(
        ["lloyd_method_dim_1_cpu", "lloyd_method_dim_1_pytorch_cpu", "lloyd_method_dim_1_pytorch_cuda"],
        ["numpy_cpu", "pytorch_cpu", "pytorch_cuda"],
        inplace=True
    )
    df_results["elapsed_time_by_iter"] = df_results.loc[:, "elapsed_time"] / df_results.loc[:, "nbr_iter"]
    df_results = df_results[df_results.nbr_iter == 100]
    df_results.drop(labels=["method_name", "device", "nbr_iter", "elapsed_time"], axis=1, inplace=True)

    df_grouped = df_results.groupby(['N', 'M', 'method'])['elapsed_time_by_iter'].mean()
    df_grouped = df_grouped.reset_index()
    df_grouped.to_csv(os.path.join(directory_path, "grouped_final_results_lloyd.csv"))

    plot_results_lloyd(df_grouped=df_grouped, M=100000, directory_path=directory_path)
    plot_results_lloyd(df_grouped=df_grouped, M=200000, directory_path=directory_path)
    plot_results_lloyd(df_grouped=df_grouped, M=500000, directory_path=directory_path)
    plot_results_lloyd(df_grouped=df_grouped, M=1000000, directory_path=directory_path)

    plot_ratios_lloyd(df_grouped=df_grouped, M=100000, directory_path=directory_path)
    plot_ratios_lloyd(df_grouped=df_grouped, M=200000, directory_path=directory_path)
    plot_ratios_lloyd(df_grouped=df_grouped, M=500000, directory_path=directory_path)
    plot_ratios_lloyd(df_grouped=df_grouped, M=1000000, directory_path=directory_path)
