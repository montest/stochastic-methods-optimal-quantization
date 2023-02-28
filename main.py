from clvq import clvq_method
from lloyd import lloyd_method, lloyd_method_optim, lloyd_method_optim_bis

if __name__ == "__main__":
    # Size of the optimal we want to build
    N = 10

    # Number of iterations
    nbr_iter = 500

    Ms = [5000]
    # Ms = [5000, 10000, 20000]
    for M in Ms:
        centroids, probas, distortion = lloyd_method_optim_bis(N=50, M=100000, nbr_iter=nbr_iter, dim=1)
        print()
        print(centroids)
        print(probas)
        print(distortion)
        # centroids, probas, distortion = lloyd_method_optim_bis(N=N, M=M, nbr_iter=nbr_iter, dim=2)
        # print()
        # print(centroids)
        # print(probas)
        # print(distortion)
        # centroids, probas, distortion = lloyd_method_optim(N=N, M=M, nbr_iter=nbr_iter)
        # print()
        # print(centroids)
        # print(probas)
        # print(distortion)
        # centroids, probas, distortion = lloyd_method(N=N, M=M, nbr_iter=nbr_iter)
        # print()
        # print(centroids)
        # print(probas)
        # print(distortion)
        # clvq_method(N=N, n=M*nbr_iter, nbr_iter=nbr_iter)
