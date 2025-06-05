from enum import IntEnum, auto
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


@torch.jit.script
def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B, S = idx.shape
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(B, 1).repeat(1, S)
    new_points = points[batch_indices, idx, :]
    return new_points


@torch.jit.script
def farthest_point_sample(xyz, npoint: int):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.zeros(B, dtype=torch.long).to(device)
    # TODO: it's canonical to start with a random point, but this causes issues with ONNX export. We can consider
    #   something like torch.arange(B, dtype=torch.long) % N which combined with batch shuffling and object shuffling
    #   could give a good amount of randomness
    # farthest = torch.randint(N, size=[B], dtype=torch.long).to(device)
    for i in range(npoint):
        idx = torch.full((B, 1), i, dtype=torch.long).to(device)
        centroids = torch.scatter(centroids, 1, idx, farthest.view(B, 1))
        centroid = torch.gather(xyz, 1, farthest.view(B, 1, 1).expand(B, 1, 3))
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.where(dist < distance, dist, distance)
        farthest = torch.max(distance, -1)[1]
    return centroids


@torch.jit.script
def query_ball_point(radius, nsample: int, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    # We build the indices as floats because the tensorRT runtime only implements the TopK operator for floats
    group_idx = torch.arange(N, dtype=torch.float32).to(device).view(1, 1, N).repeat(B, S, 1)
    sqrdists = square_distance(new_xyz, xyz)
    if radius.dim() == 0:
        # radius is 0-D: radius_sq is also 0-D, so it broadcasts against (B,S,N)
        radius_sq = radius * radius
    else:
        # radius.dim() == 1, and radius.size(0) == B
        # We reshape to (B, 1, 1) so it broadcasts along S and N.
        radius_sq = (radius * radius).view(B, 1, 1)
    group_idx[sqrdists > radius_sq] = N
    group_idx = torch.topk(group_idx, k=nsample, dim=-1, largest=False)[0]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    group_idx = torch.where(group_idx == N, group_first, group_idx)
    return group_idx.to(torch.long)


@torch.jit.script
def sample_and_group(npoint: int, radius, nsample: int, xyz, points):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    _, _, D = points.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, S, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx.reshape(B, npoint * nsample)).reshape(B, npoint, nsample, C)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_points = index_points(points, idx.reshape(B, npoint * nsample)).reshape(B, npoint, nsample, D)
    new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    return new_xyz, new_points


@torch.jit.script
def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    npoint: int
    radius: float
    nsample: int

    def __init__(self, npoint=0, radius=0.0, nsample=0, in_channel=3, mlp=(), group_all: bool = False,
                 radius_absolute: bool = False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.radius_absolute = radius_absolute
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, stride=1, padding=0))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel, momentum=0.1))
            last_channel = out_channel
        self.out_channel = last_channel
        self.group_all = group_all

    def forward(self, xyz, points, max_var_per_batch):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points feature data, [B, N, D]
            max_var_per_batch: largest variance of object along the three axes
        Return:
            xyz: sampled points position data, [B, S, C']
            points: sample points feature data, [B, S, D']
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            if not self.radius_absolute:
                radius = self.radius * max_var_per_batch
            else:
                radius = self.radius
            new_xyz, new_points = sample_and_group(self.npoint, radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]
        for bn, conv in zip(self.mlp_bns, self.mlp_convs):
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]
        return new_xyz, new_points.permute(0, 2, 1)


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class Transformer(nn.Module):
    def fit(self, data, mask=None):
        raise NotImplementedError

    def forward(self, data, mask=None):
        raise NotImplementedError


class Transform(nn.Module):
    class Type(IntEnum):
        DROP = auto()
        NONE = auto()
        Z_SCORE = auto()
        MEAN_SUBTRACT = auto()

    def __init__(self, num_dimensions: int, types=None):
        super().__init__()
        # Default treatment is to forward all input dimensions as-is
        if types is None:
            types = [self.Type.NONE] * num_dimensions
        assert len(types) == num_dimensions
        self.register_buffer('num_dimensions', torch.tensor(num_dimensions))
        self.register_buffer('types', torch.tensor(types))

        # Create transformers
        self.transformers = nn.ModuleDict()
        for dim, transform_type in enumerate(types):
            if module := self.make_module(transform_type):
                self.transformers.add_module(str(dim), module)

    @property
    def num_dimensions_transformed(self):
        return len(self.transformers)

    def make_module(self, transform_type: Type) -> Transformer | None:
        match transform_type:
            case self.Type.NONE:
                return Identity()
            case self.Type.Z_SCORE:
                return ZScorer()
            case self.Type.MEAN_SUBTRACT:
                return MeanSubtractor()

    def fit_dim(self, data, dim, mask=None):
        try:
            transformer = self.transformers[str(dim)]
        except KeyError:
            pass
        else:
            transformer.fit(data[..., dim], mask)

    def fit(self, data, mask=None):
        for dim in range(self.num_dimensions):
            self.fit_dim(data, dim, mask)

    def forward_dim(self, data, dim, mask=None):
        try:
            transformer = self.transformers[str(dim)]
        except KeyError:
            pass
        else:
            return transformer(data[..., dim], mask)

    def forward(self, data, mask=None):
        data_transformed = [self.forward_dim(data, dim, mask) for dim in range(self.num_dimensions)]
        # At this stage we possibly drop some dimensions
        return torch.stack([data_dim for data_dim in data_transformed if data_dim is not None], dim=-1)


class Identity(Transformer):
    def fit(self, data, mask=None):
        pass

    def forward(self, data, mask=None):
        return data


class MeanSubtractor(Transformer):
    def fit(self, data, mask=None):
        pass

    def forward(self, data, mask=None):
        if mask is None:
            return data - torch.mean(data, dim=-1, keepdim=True)
        # Mean of masked elements in each row
        masked_sum = torch.sum(data * mask, dim=-1, keepdim=True)
        masked_count = torch.sum(mask, dim=-1, keepdim=True)
        masked_mean = masked_sum / masked_count
        return torch.where(mask, data - masked_mean, 0.0)


class ZScorer(Transformer):
    mean: torch.Tensor
    var: torch.Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.tensor(0.0))
        self.register_buffer('var', torch.tensor(1.0))

    def fit(self, data: torch.Tensor, mask=None):
        if mask is not None:
            data = data[mask]
        self.mean = torch.nanmean(data)
        self.var = (data - self.mean).square().nanmean()

    def forward(self, data: torch.Tensor, mask=None):
        data = (data - self.mean.to(data.device)) / torch.sqrt(self.var).to(data.device)
        if mask is not None:
            return torch.where(mask, data, 0.0)
        else:
            return data


class UnitSphereNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        """
        A layer that:
          1) Extracts the first 3 channels of `data` as (x,y,z) coordinates.
          2) Centers those coordinates and scales them so that
             the farthest valid point in each cloud lies on the unit sphere.
          3) Leaves any additional feature channels (beyond the first 3) unchanged,
             except they are zeroed out where mask == 0 (if mask is provided).

        Args:
            eps (float): Small epsilon to avoid division by zero if all points collapse.
        """
        super().__init__()
        self.eps = eps

    def forward(self, data: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            data:  Tensor of shape (B, N, F), where F >= 3 (Features).
                   The first 3 channels are (x, y, z). The remaining (F - 3) channels
                   are other per-point features (rcs, snr_db, velocity) and remain untouched.
            mask:  Optional boolean or float tensor of shape (B, N) or (B, N, 1).
                   Indicates which points are valid (1) vs. invalid (0).
                   If None, all points are treated as valid.

        Returns:
            Tensor of shape (B, N, F). For each batch element and each of the N points:
              - Channels [:, :, :3] contain the centered & unit-sphere‐scaled coordinates.
              - Channels [:, :, 3:] are unchanged, except zeroed out where mask == 0.
              - If mask is provided, any point with mask == 0 will have all F features zeroed.
        """
        B, N, F = data.shape
        assert F >= 3, "data.shape[-1] must be at least 3 (for x, y, z)."

        # Split coordinates vs. features
        coords = data[..., :3]  # shape (B, N, 3)
        features = data[..., 3:]  # shape (B, N, F - 3), may be empty if F == 3

        # If no mask was provided, create one by checking for all-zero rows:
        if mask is None:
            mask_bool = torch.any(data != 0, dim=-1, keepdim=True)  # → (B, N, 1)
            mask = mask_bool.float()  # convert to float

        # Ensure mask has right dimension
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)  # → (B, N, 1)

        # Split coords vs. extras
        coords = data[..., :3]  # (B, N, 3)
        extras = data[..., 3:]  # (B, N, F-3) or empty if F == 3

        # Compute masked centroid of the first 3 dims:
        masked_sum = torch.sum(coords * mask, dim=1, keepdim=True)  # (B, 1, 3)
        masked_count = torch.sum(mask, dim=1, keepdim=True)  # (B, 1, 1)
        masked_count = masked_count.clamp(min=self.eps)  # avoid /0
        centroid = masked_sum / masked_count  # (B, 1, 3)

        # Subtract centroid from all coords
        centered = coords - centroid  # (B, N, 3)

        # Zero out invalid points before computing distances -> avoid comp overhead
        centered_masked = centered * mask  # (B, N, 3)

        # Radial distance of each valid point to origin
        dists = torch.norm(centered_masked, p=2, dim=2)  # (B, N)

        # Max distance per batch element
        max_dist, _ = torch.max(dists, dim=1, keepdim=True)  # (B, 1)
        max_dist = max_dist.clamp(min=self.eps)  # avoid /0

        # Scale all centered coords by that radius
        scaled = centered / max_dist.unsqueeze(-1)  # (B, N, 3)

        # Zero out any invalid coords
        normalized_coords = scaled * mask  # (B, N, 3)

        # Also zero out extras wherever mask == 0
        out_extras = extras * mask  # (B, N, F-3) or empty

        # 9) Re‐assemble full feature tensor
        if F == 3:
            return normalized_coords
        else:
            return torch.cat([normalized_coords, out_extras], dim=-1)


class UnitCubeNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        """
        A layer that:
          1) Extracts the first 3 channels of `data` as (x,y,z) coordinates.
          2) Centers those coordinates (optionally masked) and then scales each axis
             independently so that, for each cloud, the maximum absolute value along x, y,
             and z becomes 1. This effectively fits the points into the axis-aligned cube
             [-1, 1]^3.
          3) Leaves any additional feature channels (beyond the first 3) unchanged,
             except they are zeroed out where mask == 0 (if mask is provided).

        Args:
            eps (float): Small epsilon to avoid division by zero if all points collapse
                         or if an axis has zero dynamic range.
        """
        super().__init__()
        self.eps = eps

    def forward(self, data: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            data:  Tensor of shape (B, N, F), where F >= 3 (Features).
                   The first 3 channels are (x, y, z). The remaining (F - 3) channels
                   are other per-point features (e.g., rcs, snr_db, velocity) and remain untouched.
            mask:  Optional boolean or float tensor of shape (B, N) or (B, N, 1).
                   Indicates which points are valid (1) vs. invalid (0).
                   If None, all points are treated as valid.

        Returns:
            Tensor of shape (B, N, F). For each batch element and each of the N points:
              - Channels [:, :, :3] contain the centered & unit-cube‐scaled coordinates.
              - Channels [:, :, 3:] are unchanged, except zeroed out where mask == 0.
              - If mask is provided, any point with mask == 0 will have all F features zeroed.
        """
        B, N, F = data.shape
        assert F >= 3, "data.shape[-1] must be at least 3 (for x, y, z)."

        # Split coordinates vs. features
        coords = data[..., :3]  # shape (B, N, 3)
        extras = data[..., 3:]  # shape (B, N, F - 3), may be empty if F == 3

        # If no mask was provided, create one by checking for all-zero rows
        if mask is None:
            mask_bool = torch.any(data != 0, dim=-1, keepdim=True)  # → (B, N, 1)
            mask = mask_bool.float()

        # Ensure mask has shape (B, N, 1)
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)  # → (B, N, 1)
        else:
            assert mask.shape[2] == 1, "mask must have shape (B, N) or (B, N, 1)"

        # 1) Compute masked centroid of the first 3 dims:
        masked_sum = torch.sum(coords * mask, dim=1, keepdim=True)  # → (B, 1, 3)
        masked_count = torch.sum(mask, dim=1, keepdim=True)  # → (B, 1, 1)
        masked_count = masked_count.clamp(min=self.eps)  # avoid divide-by-zero
        centroid = masked_sum / masked_count  # → (B, 1, 3)

        # 2) Subtract centroid from all coords
        centered = coords - centroid  # → (B, N, 3)

        # 3) Zero out invalid points before computing per-axis max
        centered_masked = centered * mask  # → (B, N, 3)

        # 4) For each axis, find the maximum absolute coordinate among valid points:
        #    abs_centered_masked has shape (B, N, 3). We want max over N → (B, 3).
        max_abs_per_axis, _ = torch.max(torch.abs(centered_masked), dim=1)  # → (B, 3)
        max_abs_per_axis = max_abs_per_axis.clamp(min=self.eps)  # avoid /0

        # 5) Scale each axis independently by its max_abs to fit into [-1, 1]
        #    Reshape to (B, 1, 3) to broadcast over N:
        scale = max_abs_per_axis.unsqueeze(1)  # → (B, 1, 3)
        scaled = centered / scale  # → (B, N, 3)

        # 6) Zero out invalid coords
        normalized_coords = scaled * mask  # → (B, N, 3)

        # 7) Also zero out extras wherever mask == 0
        out_extras = extras * mask  # → (B, N, F - 3) or empty

        # 8) Re‐assemble full feature tensor
        if F == 3:
            return normalized_coords
        else:
            return torch.cat([normalized_coords, out_extras], dim=-1)
