import sys

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange

np.set_printoptions(threshold=np.inf, linewidth=10_000)
torch.set_printoptions(profile="full", linewidth=10_000)


def lloyd_method_pytorch(N: int, M: int, nbr_iter: int, device: str, closest_centroid_method: int=1, seed: int = 0):
    """
    Apply `nbr_iter` iterations of the Randomized Lloyd algorithm in order to build an optimal quantizer of size `N`
    for a Gaussian random variable.

    N: number of centroids
    M: number of samples to generate
    nbr_iter: number of iterations of fixed point search
    device: device on which perform the computations: "cuda" or "cpu"
    seed: torch seed for reproducibility

    Returns: centroids and probabilities associated to each centroid
    """
    torch.manual_seed(seed=seed)
    with torch.no_grad():

        mean, sigma = 0, 1
        xs = torch.tensor(torch.randn(M) * sigma + mean, dtype=torch.float32)

        # Initialize quantization centroids randomly
        centroids = torch.randn(N) * sigma + mean
        centroids, indices = centroids.sort()

        xs = xs.to(device)
        x_reshaped = xs.reshape(M, 1, 1)
        x_reshaped = x_reshaped.to(device)

        centroids = centroids.to(device)
        # Repeat the quantization process until convergence
        with trange(nbr_iter, desc=f'Lloyd method (pytorch: {device})') as t:
            for step in t:
                if closest_centroid_method == 1:
                    # slow version
                    # dist_centroids_points = torch.norm(centroids - xs.view(M, 1, 1), dim=1)
                    dist_centroids_points = torch.norm(centroids - x_reshaped, dim=1)
                    indices = dist_centroids_points.argmin(dim=1)
                elif closest_centroid_method == 2:
                    # quick version
                    # Compute the vertices that separate the quantization levels
                    vertices = 0.5 * (centroids[:-1] + centroids[1:])
                    indices = torch.sum(xs[:, None] >= vertices[None, :], dim=1).long()
                else:
                    raise ValueError(f"Wrong value for closest_centroid_method in method lloyd_method")

                # Compute the new quantization levels as the mean of the samples assigned to each level
                centroids = torch.tensor([torch.mean(xs[indices == i]) for i in range(N)]).to(device)

                # Check if the quantization levels have converged
                # if torch.allclose(centroids, new_quantization_levels, rtol=epsilon):
                #     break

                # Update the quantization levels
                # centroids = new_quantization_levels

                # compute probas
                # probabilities = torch.bincount(indices).numpy() / float(M)
                # probabilities = np.array([xs[indices == i].size()[0] for i in range(N)])/float(M)

                # compute distortion
                # quantized_signal = centroids[indices]
                # distortion = torch.mean((xs - quantized_signal)**2) * 0.5
                # t.set_postfix(distortion=distortion.item())

                # xs = torch.randn(n) * sigma + mean
                # xs = torch.tensor(xs, dtype=torch.float32)

        # Compute the probability of each centroid
        vertices = 0.5 * (centroids[:-1] + centroids[1:])
        indices = torch.sum(xs[:, None] >= vertices[None, :], dim=1).long()
        # probabilities = np.array([xs[indices == i].size()[0] for i in range(N)])/float(M)
        probabilities = torch.bincount(indices).to('cpu').numpy()/float(M)

        return centroids.to('cpu').numpy(), probabilities


def lloyd_method_dim_1_pytorch(N: int, M: int, nbr_iter: int, device: str, seed: int = 0):
    """
    Apply `nbr_iter` iterations of the Randomized Lloyd algorithm in order to build an optimal quantizer of size `N`
    for a Gaussian random variable. This implementation is done using torch.

    N: number of centroids
    M: number of samples to generate
    nbr_iter: number of iterations of fixed point search
    device: device on which perform the computations: "cuda" or "cpu"
    seed: torch seed for reproducibility

    Returns: centroids, probabilities associated to each centroid and distortion
    """
    torch.manual_seed(seed=seed)  # Set seed in order to be able to reproduce the results

    with torch.no_grad():
        # Draw M samples of gaussian variable
        xs = torch.randn(M)
        # xs = torch.tensor(torch.randn(M), dtype=torch.float32)
        xs = xs.to(device)  # send samples to correct device

        # Initialize the Voronoi Quantizer randomly
        centroids = torch.randn(N)
        centroids, index = centroids.sort()
        centroids = centroids.to(device)  # send centroids to correct device

        with trange(nbr_iter, desc=f'Lloyd method (pytorch: {device})') as t:
            for step in t:
                # Compute the vertices that separate the centroids
                vertices = 0.5 * (centroids[:-1] + centroids[1:])

                # Find the index of the centroid that is closest to each sample
                index_closest_centroid = torch.sum(xs[:, None] >= vertices[None, :], dim=1).long()

                # Compute the new quantization levels as the mean of the samples assigned to each level
                centroids = torch.tensor([torch.mean(xs[index_closest_centroid == i]) for i in range(N)]).to(device)

        # Compute, for each sample, the distance to each centroid
        dist_centroids_points = torch.norm(centroids - xs.reshape(M, 1, 1), dim=1)
        # Find the index of the centroid that is closest to each sample using the previously computed distances
        index_closest_centroid = dist_centroids_points.argmin(dim=1)
        # Compute the probability of each centroid
        probabilities = torch.bincount(index_closest_centroid).to('cpu').numpy()/float(M)
        # Compute the final distortion between the samples and the quantizer
        distortion = torch.mean(dist_centroids_points[torch.arange(M), index_closest_centroid] ** 2).item() * 0.5
        return centroids.to('cpu').numpy(), probabilities, distortion


# centroids, probas, distortion = lloyd_method_dim_1_pytorch(M=100000, N=50, nbr_iter=500, device='cuda')
centroids, probas, distortion = lloyd_method_dim_1_pytorch(M=100000, N=50, nbr_iter=500, device='cpu')
print(centroids)
print(probas)
print(distortion)
