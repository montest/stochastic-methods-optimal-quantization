import torch
import sys

import numpy as np


def find_closest_centroid(centroids: torch.Tensor, p: torch.Tensor):
    index_closest_centroid = -1
    min_dist = sys.float_info.max
    for i, x_i in enumerate(centroids):
        dist = np.linalg.norm(x_i - p)
        if dist < min_dist:
            index_closest_centroid = i
            min_dist = dist
    return index_closest_centroid, min_dist


if __name__ == "__main__":
    torch.manual_seed(10)
    N = 10
    dim = 2
    centroids = torch.normal(mean=torch.zeros(N, dim))
    # centroids = torch.arange(-5, 5+1, 1)
    sample = torch.normal(mean=torch.zeros(1, dim))
    # sample = torch.normal(mean=torch.zeros(1, dim))
    print(centroids)
    print(sample)
    dist = torch.norm(centroids-sample, dim=1)
    print(dist)
    index = torch.argmin(dist, dim=0)
    print(index)

    samples = torch.normal(mean=torch.zeros(3, dim))
    # sample = torch.normal(mean=torch.zeros(1, dim))
    print(centroids)
    print(samples)
    dist = torch.norm(centroids-samples, dim=1)
    print(dist)
    index = torch.argmin(dist, dim=0)
    print(index)
