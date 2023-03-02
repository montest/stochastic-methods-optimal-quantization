import sys

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange

np.set_printoptions(threshold=np.inf, linewidth=10_000)
torch.set_printoptions(profile="full", linewidth=10_000)


def lloyd_method(N: int, M: int, nbr_iter: int, device:str):
    """
    Perform scalar quantization using the Lloyd algorithm for a Gaussian random variable.

    x: input signal
    k: number of quantization levels

    Returns: quantized signal and quantization levels
    """
    with torch.no_grad():

      mean, sigma = 0, 1
      x = torch.tensor(torch.randn(M) * sigma + mean, dtype=torch.float32)

      # Initialize quantization centroids randomly
      centroids = torch.randn(N) * sigma + mean
      print(centroids)
      centroids, indices = centroids.sort()
      print(centroids)

      x = x.to(device)
      centroids = centroids.to(device)
      # Repeat the quantization process until convergence
      with trange(nbr_iter, desc=f'Lloyd method (pytorch: {device})') as t:
          for step in t:
              # x = torch.tensor(torch.randn(M) * sigma + mean, dtype=torch.float32)
              # # quick method
              # # Compute the vertices that separate the quantization levels
              vertices = 0.5 * (centroids[:-1] + centroids[1:])

              #
              # # Assign each sample to the closest quantization level
              indices = torch.sum(x[:, None] >= vertices[None, :], dim=1).long()

              # slow one
              #dist_centroids_points = torch.norm(centroids - x.view(M, 1, 1), dim=1)
              #indices = dist_centroids_points.argmin(dim=1)

              # Compute the new quantization levels as the mean of the samples assigned to each level
              centroids = torch.tensor([torch.mean(x[indices == i]) for i in range(N)]).to(device)

              # Check if the quantization levels have converged
              # if torch.allclose(centroids, new_quantization_levels, rtol=epsilon):
              #     break

              # Update the quantization levels
              # centroids = new_quantization_levels

              # compute probas
              # probabilities = torch.bincount(indices).numpy() / float(M)
              # probabilities = np.array([x[indices == i].size()[0] for i in range(N)])/float(M)

              # compute distortion
              # quantized_signal = centroids[indices]
              # distortion = torch.mean((x - quantized_signal)**2) * 0.5
              # t.set_postfix(distortion=distortion.item())

              # x = torch.randn(n) * sigma + mean
              # x = torch.tensor(x, dtype=torch.float32)

      # Compute the probability of each centroid
      vertices = 0.5 * (centroids[:-1] + centroids[1:])
      indices = torch.sum(x[:, None] >= vertices[None, :], dim=1).long()
      # probabilities = np.array([x[indices == i].size()[0] for i in range(N)])/float(M)
      probabilities = torch.bincount(indices).to('cpu').numpy()/float(M)

      return centroids.to('cpu').numpy(), probabilities


# Generate a sample of a Gaussian random variable
torch.manual_seed(0)

# Apply the Lloyd-Max algorithm with 4 quantization levels
with torch.no_grad():
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    centroids, probas = lloyd_method(M=100000, N=50, nbr_iter=500, device='cuda')
    centroids, probas = lloyd_method(M=100000, N=50, nbr_iter=500, device='cpu')
print(centroids)
print(probas)
print(probas.sum())

# Compute the mean squared error between the original signal and the quantized signal

