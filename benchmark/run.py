from lloyd_optim import lloyd_method_dim_1
from lloyd_torch import lloyd_method_dim_1_pytorch
from benchmark.utils import testing_method


path_to_results = "results.csv"

basic_parameters_grid = {
    "M": [100000, 200000, 500000, 1000000],
    "seed": [0, 1, 2],
    "N": [10, 20, 50, 100, 200, 500],
    "nbr_iter": [100],
}
pytorch_parameters_grid = basic_parameters_grid.copy()
pytorch_parameters_grid["device"] = ["cuda", "cpu"]

testing_method(lloyd_method_dim_1_pytorch, pytorch_parameters_grid, path_to_results)
testing_method(lloyd_method_dim_1, basic_parameters_grid, path_to_results)
