import numpy as np

from tqdm import trange
from utils import get_probabilities_and_distortion

np.set_printoptions(threshold=np.inf, linewidth=10_000)


def lr(N: int, n: int):
    a = 4.0 * N
    b = np.pi ** 2 / float(N * N)
    return a / float(a + b * (n+1.))


def clvq_method_dim_1(N: int, M: int, num_epochs: int, seed: int = 0):
    """
    Apply `nbr_iter` iterations of the Competitive Learning Vector Quantization algorithm in order to build an optimal
     quantizer of size `N` for a Gaussian random variable. This implementation is done using numpy.

    N: number of centroids
    M: number of samples to generate
    num_epochs: number of epochs of fixed point search
    seed: numpy seed for reproducibility

    Returns: centroids, probabilities associated to each centroid and distortion
    """
    np.random.seed(seed)  # Set seed in order to be able to reproduce the results

    # Draw M samples of gaussian variable
    xs = np.random.normal(0, 1, size=M)

    # Initialize the Voronoi Quantizer randomly and sort it
    centroids = np.random.normal(0, 1, size=N)
    centroids.sort(axis=0)

    with trange(num_epochs, desc=f'CLVQ method - N: {N} - M: {M} - seed: {seed} (numpy)') as epochs:
        for epoch in epochs:
            for step in range(M):
                # Compute the vertices that separate the centroids
                vertices = 0.5 * (centroids[:-1] + centroids[1:])

                # Find the index of the centroid that is closest to each sample
                index_closest_centroid = np.sum(xs[step, None] >= vertices[None, :])

                gamma_n = 1e-2
                # gamma_n = lr(N, epoch*M + step)

                # Update the closest centroid using the local gradient
                centroids[index_closest_centroid] = centroids[index_closest_centroid] - gamma_n * (centroids[index_closest_centroid] - xs[step])

    probabilities, distortion = get_probabilities_and_distortion(centroids, xs)
    return centroids, probabilities, distortion



########################################################################
####### Old code where probas and distortion are computed inline #######
########################################################################

# def clvq_method_dim_1(N: int, M: int, num_epochs: int, seed: int = 0):
#     """
#     Apply `nbr_iter` iterations of the Competitive Learning Vector Quantization algorithm in order to build an optimal
#      quantizer of size `N` for a Gaussian random variable. This implementation is done using numpy.
#
#     N: number of centroids
#     M: number of samples to generate
#     num_epochs: number of epochs of fixed point search
#     seed: numpy seed for reproducibility
#
#     Returns: centroids, probabilities associated to each centroid and distortion
#     """
#     probabilities = np.zeros(N)
#     distortion = 0.
#     np.random.seed(seed)  # Set seed in order to be able to reproduce the results
#
#     # Draw M samples of gaussian variable
#     xs = np.random.normal(0, 1, size=M)
#
#     # Initialize the Voronoi Quantizer randomly and sort it
#     centroids = np.random.normal(0, 1, size=N)
#     centroids.sort(axis=0)
#
#     with trange(num_epochs, desc=f'CLVQ method - N: {N} - M: {M} - seed: {seed} (numpy)') as epochs:
#         for epoch in epochs:
#             for step in range(M):
#                 # Compute the vertices that separate the centroids
#                 vertices = 0.5 * (centroids[:-1] + centroids[1:])
#
#                 # Find the index of the centroid that is closest to each sample
#                 index_closest_centroid = np.sum(xs[step, None] >= vertices[None, :])
#                 l2_dist = np.linalg.norm(centroids[index_closest_centroid] - xs[step])
#
#                 gamma_n = lr(N, epoch*M + step)
#
#                 # Update the closest centroid using the local gradient
#                 centroids[index_closest_centroid] = centroids[index_closest_centroid] - gamma_n * (centroids[index_closest_centroid] - xs[step])
#
#                 # Update the distortion using gamma_n
#                 distortion = (1 - gamma_n) * distortion + 0.5 * gamma_n * l2_dist ** 2
#
#                 # Update probabilities
#                 probabilities = (1 - gamma_n) * probabilities
#                 probabilities[index_closest_centroid] += gamma_n
#
#                 if any(np.isnan(centroids)):
#                     break
#             epochs.set_postfix(distortion=distortion)
#
#     # probabilities, distortion = get_probabilities_and_distortion(centroids, xs)
#     return centroids, probabilities, distortion
