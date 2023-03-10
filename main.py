from lloyd_optim import lloyd_method_optim, lloyd_method_dim_1
from lloyd_torch import lloyd_method_dim_1_pytorch

if __name__ == "__main__":
    # Size of the optimal we want to build
    N = 10

    # Number of iterations
    nbr_iter = 100

    Ms = [10000]
    # # Ms = [5000, 10000, 20000]
    # parameters_grid = {
    #     "N": [10, 20, 50, 100, 200, 500],
    #     "M": [5000, 10000, 20000, 100000],
    #     "nbr_iter": [10, 100, 500, 1000],
    #     "seed": [0, 1, 2, 3, 4]
    # }
    # path_to_results = "results.csv"
    #
    # testing_method(lloyd_method_dim_1, parameters_grid, path_to_results)
    #
    for M in Ms:
        print("Testing dim 1")
        centroids, probas, distortion = lloyd_method_optim(N=N, M=M, nbr_iter=nbr_iter, dim=1)
        print()
        print(centroids)
        print(probas)
        print(distortion)
        centroids, probas, distortion = lloyd_method_dim_1(N=N, M=M, nbr_iter=nbr_iter)
        print()
        print(centroids)
        print(probas)
        print(distortion)
        centroids, probas, distortion = lloyd_method_dim_1_pytorch(N=N, M=M, nbr_iter=nbr_iter, device='cpu')
        print()
        print(centroids)
        print(probas)
        print(distortion)
        # print("Testing dim 2")
    #     centroids, probas, distortion = lloyd_method_optim(N=N, M=M, nbr_iter=nbr_iter, dim=2)
    #     print()
    #     print(centroids)
    #     print(probas)
    #     print(distortion)
    #     centroids, probas, distortion = lloyd_method(N=N, M=M, nbr_iter=nbr_iter, use_optimized_version=True)
    #     print()
    #     print(centroids)
    #     print(probas)
    #     print(distortion)
    #     centroids, probas, distortion = lloyd_method(N=N, M=M, nbr_iter=nbr_iter, use_optimized_version=False)
    #     print()
    #     print(centroids)
    #     print(probas)
    #     print(distortion)
    #     # clvq_method(N=N, n=M*nbr_iter, nbr_iter=nbr_iter)
