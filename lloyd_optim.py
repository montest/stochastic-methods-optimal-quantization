import numpy as np

from tqdm import trange

np.set_printoptions(threshold=np.inf, linewidth=10_000)


def lloyd_method_dim_1(N: int, M: int, nbr_iter: int, seed: int = 0):
    """
    Apply `nbr_iter` iterations of the Randomized Lloyd algorithm in order to build an optimal quantizer of size `N`
    for a Gaussian random variable. This implementation is done using numpy.

    N: number of centroids
    M: number of samples to generate
    nbr_iter: number of iterations of fixed point search
    seed: numpy seed for reproducibility

    Returns: centroids, probabilities associated to each centroid and distortion
    """
    np.random.seed(seed)  # Set seed in order to be able to reproduce the results

    # Draw M samples of gaussian variable
    xs = np.random.normal(0, 1, size=M)

    # Initialize the Voronoi Quantizer randomly and sort it
    centroids = np.random.normal(0, 1, size=N)
    centroids.sort(axis=0)

    with trange(nbr_iter, desc='Lloyd method (numpy)') as t:
        for step in t:
            # Compute the vertices that separate the centroids
            vertices = 0.5 * (centroids[:-1] + centroids[1:])

            # Find the index of the centroid that is closest to each sample
            index_closest_centroid = np.sum(xs[:, None] >= vertices[None, :], axis=1)

            # Compute the new quantization levels as the mean of the samples assigned to each level
            centroids = np.array([np.mean(xs[index_closest_centroid == i], axis=0) for i in range(N)])

    # Compute, for each sample, the distance to each centroid
    dist_centroids_points = np.linalg.norm(centroids.reshape((N, 1)) - xs.reshape(M, 1, 1), axis=2)
    # Find the index of the centroid that is closest to each sample using the previously computed distances
    index_closest_centroid = dist_centroids_points.argmin(axis=1)
    # Compute the probability of each centroid
    probabilities = np.bincount(index_closest_centroid) / float(M)
    # Compute the final distortion between the samples and the quantizer
    distortion = np.mean(dist_centroids_points[np.arange(M), index_closest_centroid] ** 2) * 0.5
    return centroids, probabilities, distortion


def lloyd_method_optim(N: int, M: int, nbr_iter: int, dim: int = 1):
    np.random.seed(0)
    xs = np.random.normal(0, 1, size=[M, dim])  # Draw M samples of gaussian vectors
    centroids = np.random.normal(0, 1, size=N*dim)  # Initialize the Voronoi Quantizer
    if dim == 1:
        centroids.sort(axis=0)
    centroids = centroids.reshape((N, dim))

    with trange(nbr_iter, desc='Lloyd method') as t:
        for step in t:
            dist_centroids_points = np.linalg.norm(centroids - xs.reshape(M, 1, dim), axis=2)
            index_closest_centroid = dist_centroids_points.argmin(axis=1)

            # Compute the new quantization levels as the mean of the samples assigned to each level
            centroids = np.array([np.mean(xs[index_closest_centroid == i], axis=0) for i in range(N)])

    dist_centroids_points = np.linalg.norm(centroids - xs.reshape(M, 1, dim), axis=2)
    index_closest_centroid = dist_centroids_points.argmin(axis=1)
    # local_count = np.array([xs[index_closest_centroid == i].shape[0] for i in range(N)])
    probabilities = np.bincount(index_closest_centroid) / float(M)
    distortion = np.mean(dist_centroids_points[np.arange(M), index_closest_centroid] ** 2) * 0.5
    if dim == 1:
        centroids = centroids.flatten()
    return centroids, probabilities, distortion


# def lloyd_method_dim_1(N: int, M: int, nbr_iter: int, seed: int = 0, closest_centroid_method: int = 1):
#     np.random.seed(seed)  # Set seed in order to be able to reproduce the results
#
#     # Draw M samples of gaussian variable
#     xs = np.random.normal(0, 1, size=M)
#
#     # Initialize the Voronoi Quantizer randomly
#     centroids = np.random.normal(0, 1, size=N)
#     centroids.sort(axis=0)
#
#     with trange(nbr_iter, desc='Lloyd method (numpy)') as t:
#         for step in t:
#             if closest_centroid_method == 1:
#                 # slow version
#                 dist_centroids_points = np.linalg.norm(centroids.reshape((N, 1)) - xs.reshape(M, 1, 1), axis=2)
#                 index_closest_centroid = dist_centroids_points.argmin(axis=1)
#             elif closest_centroid_method == 2:
#                 # quick version
#                 vertices = 0.5 * (centroids[:-1] + centroids[1:])
#                 index_closest_centroid = np.sum(xs[:, None] >= vertices[None, :], axis=1)
#             else:
#                 raise ValueError(f"Wrong value for closest_centroid_method in method lloyd_method_optim_dim_1")
#
#             # Compute the new quantization levels as the mean of the samples assigned to each level
#             centroids = np.array([np.mean(xs[index_closest_centroid == i], axis=0) for i in range(N)])
#
#     dist_centroids_points = np.linalg.norm(centroids.reshape((N, 1)) - xs.reshape(M, 1, 1), axis=2)
#     index_closest_centroid = dist_centroids_points.argmin(axis=1)
#     # local_count = np.array([xs[index_closest_centroid == i].shape[0] for i in range(N)])
#     probabilities = np.bincount(index_closest_centroid) / float(M)
#     distortion = np.mean(dist_centroids_points[np.arange(M), index_closest_centroid] ** 2) * 0.5
#     return centroids, probabilities, distortion