import os

import pandas as pd

from benchmark.benchmark_utils import plot_results_clvq, plot_ratios_clvq

if __name__ == "__main__":
    results_path = "/Users/thibautmontes/GitHub/stochastic-methods-optimal-quantization/benchmark/"
    plot_path = "/Users/thibautmontes/GitHub/stochastic-methods-optimal-quantization/_output/gaussian/pytorch/clvq/"
    path_to_results = os.path.join(results_path, "final_results_clvq.csv")
    if os.path.exists(path_to_results) and os.path.getsize(path_to_results) > 0:
        df_results = pd.read_csv(path_to_results, index_col=0)
    else:
        raise Exception(f"Wrong path to benchmark results {path_to_results}")
    df_results['device'].fillna("cpu", inplace=True)
    df_results["method"] = df_results.loc[:, "method_name"] + "_" + df_results.loc[:, "device"]
    df_results["method"].replace(
        ["clvq_method_dim_1_cpu", "clvq_method_dim_1_pytorch_cpu", "clvq_method_dim_1_pytorch_cuda",
         "clvq_method_dim_1_pytorch_autograd_cpu", "clvq_method_dim_1_pytorch_autograd_cuda"],
        ["numpy_cpu", "pytorch_cpu", "pytorch_cuda", "pytorch_autograd_cpu", "pytorch_autograd_cuda"],
        inplace=True
    )
    df_results["elapsed_time_by_epoch"] = df_results.loc[:, "elapsed_time"] / df_results.loc[:, "num_epoch"]
    df_results = df_results[df_results.num_epoch == 1]
    df_results.drop(labels=["method_name", "device", "num_epoch", "elapsed_time"], axis=1, inplace=True)

    df_grouped = df_results.groupby(['N', 'M', 'method'])['elapsed_time_by_epoch'].mean()
    df_grouped = df_grouped.reset_index()
    df_grouped.to_csv(os.path.join(results_path, "grouped_final_results_clvq.csv"))

    Ms = set(df_results.M)
    for M in Ms:
        plot_results_clvq(df_grouped=df_grouped, M=M, directory_path=plot_path)
        plot_ratios_clvq(df_grouped=df_grouped, M=M, directory_path=plot_path)
