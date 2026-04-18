"""
PointNet++ low-level primitives: FPS, ball query, grouping.
All operations are pure PyTorch (no CUDA extensions required).
"""

from __future__ import annotations

import torch
from torch import Tensor


def farthest_point_sample(xyz: Tensor, npoint: int) -> Tensor:
    """
    Farthest Point Sampling.

    Args:
        xyz:    (B, N, 3)
        npoint: number of points to sample

    Returns:
        idx: (B, npoint) — indices into xyz
    """
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.zeros(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest, :].unsqueeze(1)  # (B, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(-1)  # (B, N)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = distance.argmax(-1)

    return centroids


def square_distance(src: Tensor, dst: Tensor) -> Tensor:
    """
    Compute squared pairwise distances between two point sets.

    Args:
        src: (B, N, 3)
        dst: (B, M, 3)

    Returns:
        dist: (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.bmm(src, dst.permute(0, 2, 1))
    dist += (src**2).sum(-1, keepdim=True)
    dist += (dst**2).sum(-1).unsqueeze(1)
    return dist.clamp(min=0)


def ball_query(radius: float, nsample: int, xyz: Tensor, new_xyz: Tensor) -> Tensor:
    """
    Ball query — find up to nsample neighbours within radius for each centroid.

    Args:
        radius:  search radius
        nsample: max neighbours
        xyz:     (B, N, 3) all points
        new_xyz: (B, S, 3) centroids

    Returns:
        group_idx: (B, S, nsample) — indices; padded with first found index
    """
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    dist = square_distance(new_xyz, xyz)  # (B, S, N)
    group_idx = dist.argsort(dim=-1)[..., :nsample]  # (B, S, nsample)
    # Replace out-of-radius with the closest in-radius point
    group_first = group_idx[:, :, 0:1].expand_as(group_idx)
    mask = dist.gather(-1, group_idx) > radius**2
    group_idx[mask] = group_first[mask]
    return group_idx


def index_points(points: Tensor, idx: Tensor) -> Tensor:
    """
    Gather points by index.

    Args:
        points: (B, N, C)
        idx:    (B, S) or (B, S, K)

    Returns:
        new_points: (B, S, C) or (B, S, K, C)
    """
    B = points.shape[0]
    raw_size = idx.shape
    idx = idx.reshape(B, -1)
    res = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, points.shape[-1]))
    return res.reshape(*raw_size, points.shape[-1])
