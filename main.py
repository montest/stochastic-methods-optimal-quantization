from clvq import clvq_method
from lloyd import lloyd_method

if __name__ == "__main__":
    # Size of the optimal we want to build
    N = 50

    # Number of iterations
    nbr_iter = 100

    Ms = [5000, 10000, 20000]
    for M in Ms:
        lloyd_method(N=N, M=M, nbr_iter=nbr_iter)

        clvq_method(N=N, n=M*nbr_iter, nbr_iter=nbr_iter)
