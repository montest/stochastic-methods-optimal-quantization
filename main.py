import time

import torch

from clvq_optim import clvq_method_dim_1
from clvq_pytorch import clvq_method_dim_1_pytorch, clvq_method_dim_1_pytorch_autograd, \
    clvq_method_dim_1_pytorch_autograd_batched, clvq_method_dim_1_pytorch_batched
from lloyd import lloyd_method
from lloyd_optim import lloyd_method_optim, lloyd_method_dim_1
from lloyd_torch import lloyd_method_dim_1_pytorch
import numpy as np

if __name__ == "__main__":
    # Size of the optimal we want to build
    N = 10

    # Number of iterations
    nbr_iter = 10

    Ms = [100000]

    for M in Ms:
        print("Testing dim 1")
        centroids, probas, distortion = clvq_method_dim_1_pytorch_batched(
            N=N, M=M, num_epochs=nbr_iter, device='cpu', batch_size=8
        )
        print()
        print(centroids)
        print(probas)
        print(probas.sum())
        print(distortion)
        time.sleep(1)
        centroids, probas, distortion = clvq_method_dim_1_pytorch_autograd_batched(
            N=N, M=M, num_epochs=nbr_iter, device='cpu', batch_size=8
        )
        print()
        print(centroids)
        print(probas)
        print(probas.sum())
        print(distortion)
        time.sleep(1)
        centroids, probas, distortion = clvq_method_dim_1(
            N=N, M=M, num_epochs=nbr_iter
        )
        print()
        print(centroids)
        print(probas)
        print(np.sum(probas))
        print(distortion)
        time.sleep(1)
        centroids, probas, distortion = clvq_method_dim_1_pytorch_autograd(
            N=N, M=M, num_epochs=nbr_iter, device='cpu'
        )
        print()
        print(centroids)
        print(probas)
        print(probas.sum())
        print(distortion)
        time.sleep(1)
        centroids, probas, distortion = clvq_method_dim_1_pytorch(
            N=N, M=M, num_epochs=nbr_iter, device='cpu'
        )
        print()
        print(centroids)
        print(probas)
        print(probas.sum())
        print(distortion)
        time.sleep(1)
        centroids, probas, distortion = lloyd_method_dim_1(
            N=N, M=M, nbr_iter=nbr_iter
        )
        print()
        print(centroids)
        print(probas)
        print(np.sum(probas))
        print(distortion)
        time.sleep(1)
        centroids, probas, distortion = lloyd_method_dim_1_pytorch(N=N, M=M, nbr_iter=nbr_iter, device='cpu')
        print()
        print(centroids)
        print(probas)
        print(np.sum(probas))
        print(distortion)
        # centroids, probas, distortion = lloyd_method_optim(N=N, M=M, nbr_iter=nbr_iter, dim=1)
        # print()
        # print(centroids)
        # print(probas)
        # print(distortion)
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
