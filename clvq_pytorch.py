import torch
from torch import nn
from tqdm import trange

from utils import get_probabilities_and_distortion

torch.set_printoptions(profile="full", linewidth=10_000)


def lr(N: int, n: int):
    a = 4.0 * N
    b = torch.pi ** 2 / float(N * N)
    return a / float(a + b * (n+1.))


def clvq_method_dim_1_pytorch(N: int, M: int, num_epochs: int, device: str, seed: int = 0):
    """
    Apply `nbr_iter` iterations of the Competitive Learning Vector Quantization algorithm in order to build an optimal
     quantizer of size `N` for a Gaussian random variable. This implementation is done using torch.

    N: number of centroids
    M: number of samples to generate
    num_epochs: number of epochs of fixed point search
    device: device on which perform the computations: "cuda" or "cpu"
    seed: numpy seed for reproducibility

    Returns: centroids, probabilities associated to each centroid and distortion
    """
    torch.manual_seed(seed=seed)  # Set seed in order to be able to reproduce the results
    with torch.no_grad():
        # Draw M samples of gaussian variable
        xs = torch.randn(M)
        xs = xs.to(device)  # send samples to correct device

        # Initialize the Voronoi Quantizer randomly and sort it
        centroids = torch.randn(N)
        centroids, index = centroids.sort()
        centroids = centroids.to(device)  # send centroids to correct device

        with trange(num_epochs, desc=f'CLVQ method - N: {N} - M: {M} - seed: {seed} (pytorch: {device})') as epochs:
            for epoch in epochs:
                for step in range(M):
                    # Compute the vertices that separate the centroids
                    vertices = 0.5 * (centroids[:-1] + centroids[1:])

                    # Find the index of the centroid that is closest to each sample
                    index_closest_centroid = torch.sum(xs[step, None] >= vertices[None, :]).long()

                    gamma_n = 1e-2
                    # gamma_n = lr(N, epoch*M + step)

                    # Update the closest centroid using the local gradient
                    centroids[index_closest_centroid] = centroids[index_closest_centroid] - gamma_n * (centroids[index_closest_centroid] - xs[step])

    probabilities, distortion = get_probabilities_and_distortion(centroids, xs)
    return centroids.to('cpu').numpy(), probabilities, distortion


def clvq_method_dim_1_pytorch_batched(N: int, M: int, num_epochs: int, device: str, batch_size: int, seed: int = 0):
    """
    Apply `nbr_iter` iterations of the Competitive Learning Vector Quantization algorithm in order to build an optimal
     quantizer of size `N` for a Gaussian random variable. This implementation is done using torch.

    N: number of centroids
    M: number of samples to generate
    num_epochs: number of epochs of fixed point search
    device: device on which perform the computations: "cuda" or "cpu"
    batch_size: batch size used for the approximation of \E ( gradient )
    seed: numpy seed for reproducibility

    Returns: centroids, probabilities associated to each centroid and distortion
    """
    torch.manual_seed(seed=seed)  # Set seed in order to be able to reproduce the results
    with torch.no_grad():

        # Draw M samples of gaussian variable
        xs = torch.randn(M)
        xs = xs.to(device)  # send samples to correct device

        # Initialize the Voronoi Quantizer randomly and sort it
        centroids = torch.randn(N)
        centroids, index = centroids.sort()
        centroids = centroids.to(device)  # send centroids to correct device

        with trange(num_epochs, desc=f'CLVQ method - N: {N} - M: {M} - batch size: {batch_size} - seed: {seed} (pytorch: {device})') as epochs:
            for epoch in epochs:
                steps = M // batch_size
                rest = M % batch_size
                for step in range(steps):
                    # Compute the vertices that separate the centroids
                    vertices = 0.5 * (centroids[:-1] + centroids[1:])

                    batch_indices = [step*batch_size + i for i in range(batch_size)]
                    # Find the index of the centroid that is closest to each sample in the batch
                    index_closest_centroid = torch.sum(xs[batch_indices, None] >= vertices[None, :], axis=1)

                    # Update the closest centroid using the local gradient
                    gamma_n = lr(N, epoch*M + step)
                    centroids[index_closest_centroid] = centroids[index_closest_centroid] - gamma_n * (centroids[index_closest_centroid] - xs[batch_indices]) / batch_size

        probabilities, distortion = get_probabilities_and_distortion(centroids, xs)
        return centroids.to('cpu').numpy(), probabilities, distortion


class Quantizer(nn.Module):
    def __init__(self, N, device):
        super(Quantizer, self).__init__()
        centroids = torch.randn(N)
        centroids, index = centroids.sort()
        self.centroids = nn.Parameter(centroids.clone().detach().requires_grad_(True))
        self.centroids = self.centroids.to(device)  # send centroids to correct device


def clvq_method_dim_1_pytorch_autograd(N: int, M: int, num_epochs: int, device: str, seed: int = 0):
    """
    Apply `nbr_iter` iterations of the Competitive Learning Vector Quantization algorithm in order to build an optimal
     quantizer of size `N` for a Gaussian random variable. This implementation is done using torch.

    N: number of centroids
    M: number of samples to generate
    num_epochs: number of epochs of fixed point search
    device: device on which perform the computations: "cuda" or "cpu"
    seed: numpy seed for reproducibility

    Returns: centroids, probabilities associated to each centroid and distortion
    """
    torch.manual_seed(seed=seed)  # Set seed in order to be able to reproduce the results
    # probabilities = torch.zeros(N)
    # distortion = 0.

    # Draw M samples of gaussian variable
    xs = torch.randn(M)
    xs = xs.to(device)  # send samples to correct device

    quantizer = Quantizer(N, device)
    quantizer.train()
    quantizer.zero_grad()
    optim = torch.optim.SGD(quantizer.parameters(), lr=1e-2, momentum=0)
    # optim = torch.optim.SGD(list(quantizer.parameters()), lr=1e-2, momentum=0)
    # optim = torch.optim.SGD(list(quantizer.parameters()), lr=1e-2, momentum=0.9)
    # local_centroids = quantizer.centroids.clone().detach()
    with trange(num_epochs, desc=f'CLVQ method - N: {N} - M: {M} - seed: {seed} (pytorch autograd: {device})') as epochs:
        for epoch in epochs:
            for step in range(M):
                # print(f"Step {step+1}")
                # Compute the vertices that separate the centroids
                with torch.no_grad():
                    vertices = 0.5 * (quantizer.centroids[:-1] + quantizer.centroids[1:])
                    # Find the index of the centroid that is closest to each sample
                    index_closest_centroid = torch.sum(xs[step, None] >= vertices[None, :]).long()
                # print(f"1:    {quantizer.centroids.data}")
                # if step > 0:
                #     print(f"1bis: {quantizer.centroids.grad.data}")
                optim.zero_grad()
                loss = 0.5 * (quantizer.centroids[index_closest_centroid] - xs[step])**2
                # loss = 0.5 * torch.linalg.vector_norm(quantizer.centroids[index_closest_centroid] - xs[step])**2
                #
                # with torch.no_grad():
                #     print(f"2:    {quantizer.centroids.grad.data}")
                #     print(f"3:    {local_centroids[index_closest_centroid] - xs[step]}")
                #
                loss.backward()
                optim.step()  # gradient descent
                #
                # print(f"4:    {quantizer.centroids.data}")
                # # Update the closest centroid using the local gradient
                # gamma_n = 0.01
                # # gamma_n = lr(N, epoch*M + step)
                # # local_centroids = quantizer.centroids.detach().copy()
                # with torch.no_grad():
                #     local_centroids[index_closest_centroid] = local_centroids[index_closest_centroid] - gamma_n * (local_centroids[index_closest_centroid] - xs[step])
                # print(f"5:    {local_centroids}")
    quantizer.eval()
    probabilities, distortion = get_probabilities_and_distortion(quantizer.centroids, xs)
    return quantizer.centroids.clone().detach().to('cpu').numpy(), probabilities, distortion


def clvq_method_dim_1_pytorch_autograd_batched(N: int, M: int, num_epochs: int, device: str, batch_size: int, seed: int = 0):
    """
    Apply `nbr_iter` iterations of the Competitive Learning Vector Quantization algorithm in order to build an optimal
     quantizer of size `N` for a Gaussian random variable. This implementation is done using torch.

    N: number of centroids
    M: number of samples to generate
    num_epochs: number of epochs of fixed point search
    device: device on which perform the computations: "cuda" or "cpu"
    batch_size: batch size used for the approximation of \E ( gradient )
    seed: numpy seed for reproducibility

    Returns: centroids, probabilities associated to each centroid and distortion
    """
    torch.manual_seed(seed=seed)  # Set seed in order to be able to reproduce the results
    # probabilities = torch.zeros(N)
    # distortion = 0.

    # Draw M samples of gaussian variable
    xs = torch.randn(M)
    xs = xs.to(device)  # send samples to correct device

    quantizer = Quantizer(N, device)
    quantizer.train()
    quantizer.zero_grad()
    optim = torch.optim.SGD(list(quantizer.parameters()), lr=1e-2, momentum=0)
    with trange(num_epochs, desc=f'CLVQ method - N: {N} - M: {M} - batch size: {batch_size} - seed: {seed} (pytorch autograd: {device})') as epochs:
        for epoch in epochs:
            steps = M // batch_size
            rest = M % batch_size
            for step in range(steps):
                batch_indices = [step*batch_size + i for i in range(batch_size)]
                # print(f"Step {step+1}")
                # Compute the vertices that separate the centroids
                with torch.no_grad():
                    vertices = 0.5 * (quantizer.centroids[:-1] + quantizer.centroids[1:])
                    # Find the index of the centroid that is closest to each sample
                    index_closest_centroid = torch.sum(xs[batch_indices, None] >= vertices[None, :], axis=1)
                # print(f"1:    {quantizer.centroids.data}")
                # if step > 0:
                #     print(f"1bis: {quantizer.centroids.grad.data}")
                optim.zero_grad()
                loss = 0.5 * torch.mean((quantizer.centroids[index_closest_centroid] - xs[batch_indices])**2)
                # loss = 0.5 * torch.linalg.vector_norm(torch.reshape(quantizer.centroids[index_closest_centroid] - xs[step], (batch_size,1)), dim=1)**2 / batch_size
                loss.backward()
                # with torch.no_grad():
                #     print(f"2:    {torch.sum(quantizer.centroids.grad.data)}")
                #     print(f"3:    {torch.sum(quantizer.centroids[index_closest_centroid] - xs[batch_indices])/batch_size}")

                optim.step()  # gradient descent
    quantizer.eval()
    probabilities, distortion = get_probabilities_and_distortion(quantizer.centroids, xs)
    return quantizer.centroids.clone().detach().to('cpu').numpy(), probabilities, distortion




########################################################################
####### Old code where probas and distortion are computed inline #######
########################################################################

# def clvq_method_dim_1_pytorch(N: int, M: int, num_epochs: int, device: str, seed: int = 0):
#     """
#     Apply `nbr_iter` iterations of the Competitive Learning Vector Quantization algorithm in order to build an optimal
#      quantizer of size `N` for a Gaussian random variable. This implementation is done using torch.
#
#     N: number of centroids
#     M: number of samples to generate
#     num_epochs: number of epochs of fixed point search
#     device: device on which perform the computations: "cuda" or "cpu"
#     seed: numpy seed for reproducibility
#
#     Returns: centroids, probabilities associated to each centroid and distortion
#     """
#     torch.manual_seed(seed=seed)  # Set seed in order to be able to reproduce the results
#     with torch.no_grad():
#         probabilities = torch.zeros(N)
#         distortion = 0.
#
#         # Draw M samples of gaussian variable
#         xs = torch.randn(M)
#         xs = xs.to(device)  # send samples to correct device
#
#         # Initialize the Voronoi Quantizer randomly and sort it
#         centroids = torch.randn(N)
#         centroids, index = centroids.sort()
#         centroids = centroids.to(device)  # send centroids to correct device
#
#         with trange(num_epochs, desc=f'CLVQ method - N: {N} - M: {M} - seed: {seed} (pytorch: {device})') as epochs:
#             for epoch in epochs:
#                 for step in range(M):
#                     # Compute the vertices that separate the centroids
#                     vertices = 0.5 * (centroids[:-1] + centroids[1:])
#
#                     # Find the index of the centroid that is closest to each sample
#                     index_closest_centroid = torch.sum(xs[step, None] >= vertices[None, :]).long()
#                     l2_dist = torch.norm(centroids[index_closest_centroid] - xs[step])
#
#                     gamma_n = lr(N, epoch*M + step)
#
#                     # Update the closest centroid using the local gradient
#                     centroids[index_closest_centroid] = centroids[index_closest_centroid] - gamma_n * (centroids[index_closest_centroid] - xs[step])
#
#                     # Update the distortion using gamma_n
#                     distortion = (1 - gamma_n) * distortion + 0.5 * gamma_n * l2_dist ** 2
#
#                     # Update probabilities
#                     probabilities = (1 - gamma_n) * probabilities
#                     probabilities[index_closest_centroid] += gamma_n
#
#                     if torch.isnan(centroids).any():
#                         break
#                 epochs.set_postfix(distortion=distortion.item())
#
#     # probabilities, distortion = get_probabilities_and_distortion(centroids, xs)
#     return centroids.to('cpu').numpy(), probabilities, distortion
