import numpy as np

from tqdm import trange
from typing import List
from utils import find_closest_centroid, Point, make_gif, get_directory, save_results


def lr(N: int, n: int):
    a = 4.0 * N
    b = np.pi ** 2 / float(N * N)
    return a / float(a + b * (n+1.))


def apply_M_gradient_descend_steps(centroids: List[Point], xs: List[Point], count: List[float], distortion: float, init_n: int, use_optimized_version: bool):
    N = len(centroids)  # Size of the quantizer

    # M steps of the Stochastic Gradient Descent
    for n, x in enumerate(xs):
        gamma_n = lr(N, init_n+n)

        # find the centroid which is the closest to sample x
        index, l2_dist = find_closest_centroid(centroids, x, use_optimized_version=use_optimized_version)

        # Update the closest centroid using the local gradient
        centroids[index] = centroids[index] - gamma_n * (centroids[index]-x)

        # Update the distortion using gamma_n
        distortion = (1 - gamma_n) * distortion + 0.5 * gamma_n * l2_dist ** 2

        # Update counter used for computing the probabilities
        count[index] = count[index] + 1

    return centroids, count, distortion


def clvq_method(N: int, n: int, nbr_iter: int, use_optimized_version: bool = True):
    if n % nbr_iter != 0:
        raise ValueError(f"nbr_iter {nbr_iter} should be a multiple of n {n}!!")
    M = int(n / nbr_iter)

    # Initialization step
    np.random.seed(0)
    xs = np.random.normal(0, 1, size=[M, 2])  # Draw M samples of gaussian vectors
    centroids = np.random.normal(0, 1, size=[N, 2])

    count = np.zeros(N)
    distortion = 0.

    with trange(nbr_iter, desc='CLVQ method') as t:
        for step in t:
            # xs = np.random.normal(0, 1, size=[M, 2])  # Draw M samples of gaussian vectors

            centroids, count, distortion = apply_M_gradient_descend_steps(centroids, xs, count, distortion, init_n=step*M, use_optimized_version=use_optimized_version)
            probas = count / np.sum(count)
            t.set_postfix(distortion=distortion, nbr_gradient_iter=(step+1)*M)

            # This is only useful when plotting the results
            # save_results(centroids, probas, distortion, step, M, method='clvq')

    # make_gif(get_directory(N, M, method='clvq'))
    return centroids, probas, distortion
