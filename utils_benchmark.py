import math
import os
import time
import itertools
import torch
import pandas as pd


def check_existance(dict_of_values, df):
    v = df.iloc[:, 0] == df.iloc[:, 0]
    for key, value in dict_of_values.items():
        v &= (df[key] == value)
    return v.any()


def testing_method(fct_to_test, parameters_grid: dict, path_to_results: str):
    if os.path.exists(path_to_results) and os.path.getsize(path_to_results) > 0:
        df_results = pd.read_csv(path_to_results, index_col=0)
    else:
        df_results = pd.DataFrame()

    keys, values = zip(*parameters_grid.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for permutations in permutations_dicts:
        dict_result = permutations.copy()
        dict_result["method_name"] = fct_to_test.__name__
        if len(df_results) > 0 and check_existance(dict_result, df_results):
            print(f"Skipping {dict_result}")
            continue
        torch.cuda.empty_cache()

        start_time = time.time()
        centroids, probabilities, distortion = fct_to_test(**permutations)
        if math.isnan(distortion):
            print(f"Results for following values {dict_result} were not saved "
                  f"because an nan was present in the centroids")
            continue
        elapsed_time = time.time() - start_time
        dict_result["elapsed_time"] = elapsed_time
        df_results = pd.concat(
            [df_results, pd.DataFrame(dict_result, index=[0])],
            ignore_index=True
        )
        df_results.to_csv(path_to_results)
