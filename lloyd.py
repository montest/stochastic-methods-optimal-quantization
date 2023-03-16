import numpy as np

from tqdm import trange
from typing import List
from utils import find_closest_centroid, Point, make_gif, get_directory, save_results

np.set_printoptions(threshold=np.inf, linewidth=10_000)


def fixed_point_iteration(centroids, xs: List[Point], use_optimized_version: bool):
    N = len(centroids)  # Size of the quantizer
    M = len(xs)  # Number of samples

    # Initialization step
    local_mean = np.zeros((N, 2))
    local_count = np.zeros(N)
    local_dist = 0.

    for x in xs:
        # find the centroid which is the closest to sample x
        index, l2_dist = find_closest_centroid(centroids, x, use_optimized_version=use_optimized_version)

        # Compute local mean, proba and distortion
        local_mean[index] = local_mean[index] + x
        local_dist += l2_dist ** 2  # Computing distortion
        local_count[index] += 1  # Count number of samples falling in cell 'index'

    for i in range(N):
        centroids[i] = local_mean[i] / local_count[i] if local_count[i] > 0 else centroids[i]

    probas = local_count / float(M)
    distortion = local_dist / float(2*M)

    return centroids, probas, distortion


def lloyd_method(N: int, M: int, nbr_iter: int, use_optimized_version:bool = True):
    np.random.seed(0)
    xs = np.random.normal(0, 1, size=[M, 2])  # Draw M samples of gaussian vectors
    centroids = np.random.normal(0, 1, size=[N, 2])  # Initialize the Voronoi Quantizer

    with trange(nbr_iter, desc='Lloyd method') as t:
        for step in t:

            # xs = np.random.normal(0, 1, size=[M, 2])  # Draw M samples of gaussian vectors

            centroids, probas, distortion = fixed_point_iteration(centroids, xs, use_optimized_version)  # Apply fixed-point search iteration
            t.set_postfix(distortion=distortion)

            # This is only useful when plotting the results
            # save_results(centroids, probas, distortion, step, M, method='lloyd')

    # make_gif(get_directory(N, M, method='lloyd'))
    return centroids, probas, distortion
