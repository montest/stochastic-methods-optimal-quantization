from lloyd_optim import lloyd_method_dim_1
from lloyd_pytorch import lloyd_method_dim_1_pytorch
from benchmark.utils import testing_method


# first warm-up in order to prepare the gpu and cpu
path_to_results = "warm_up_results_lloyd.csv"

warm_up_basic_parameters_grid = {
    "M": [500000],
    "seed": [0],
    "N": [20],
    "nbr_iter": [100],
}
warm_up_pytorch_parameters_grid = warm_up_basic_parameters_grid.copy()
warm_up_pytorch_parameters_grid["device"] = ["cuda", "cpu"]

testing_method(lloyd_method_dim_1_pytorch, warm_up_pytorch_parameters_grid, path_to_results)
testing_method(lloyd_method_dim_1, warm_up_basic_parameters_grid, path_to_results)


# then the true benchmark starts
path_to_results = "final_results_lloyd.csv"

basic_parameters_grid = {
    "M": [200000, 500000, 1000000],
    "seed": [0, 1, 2],
    "N": [10, 20, 50, 100, 200, 500],
    "nbr_iter": [100],
}
pytorch_parameters_grid = basic_parameters_grid.copy()
pytorch_parameters_grid["device"] = ["cuda", "cpu"]

testing_method(lloyd_method_dim_1_pytorch, pytorch_parameters_grid, path_to_results)
testing_method(lloyd_method_dim_1, basic_parameters_grid, path_to_results)
