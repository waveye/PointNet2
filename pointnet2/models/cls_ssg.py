import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2.models.utils import PointNetSetAbstraction, Transform, UnitSphereNormalization, UnitCubeNormalization


class get_model(nn.Module):
    def __init__(self, num_classes, num_dimensions=3, transform=None):
        super(get_model, self).__init__()
        self.register_buffer('num_classes', torch.tensor(num_classes))
        self.register_buffer('num_dimensions', torch.tensor(num_dimensions))
        self.transform = Transform(num_dimensions, transform)

        # Config
        self.absolute_radius = True
        # If absolute_radius=False: r1, ... are fractions of 2*max_std_variance (max of std_x, std_y, std_z) of a sample
        # If absolute_radius=True: r1, ... are absolute radii for ball query
        self.r1 = 0.25  # fine features
        self.r2 = 0.5   # local features
        self.r3 = 0.75
        self.r4 = 1.0   # features describe big parts of objects

        self.npoint1 = 100
        self.npoint2 = 80
        self.npoint3 = 60
        self.npoint4 = 20

        self.nsample1 = 10
        self.nsample2 = 10
        self.nsample3 = 10
        self.nsample4 = 10

        self.mlp1 = (16, 16, 32)
        self.mlp2 = (32, 32, 32)
        self.mlp3 = (32, 32, 64)
        self.mlp4 = (64, 64, 128)
        self.mlp_last = (128, 256, 512)

        self.fc1_out = 128
        self.fc2_out = 64
        self.fc_dropout = 0.3

        self.sa1 = PointNetSetAbstraction(
            npoint=self.npoint1, radius=self.r1, nsample=self.nsample1,
            in_channel=self.transform.num_dimensions_transformed, mlp=self.mlp1, radius_absolute=self.absolute_radius)
        self.sa2 = PointNetSetAbstraction(
            npoint=self.npoint2, radius=self.r2, nsample=self.nsample2,
            in_channel=3 + self.sa1.out_channel, mlp=self.mlp2, radius_absolute=self.absolute_radius)
        self.sa3 = PointNetSetAbstraction(
            npoint=self.npoint3, radius=self.r3, nsample=self.nsample3, in_channel=3 + self.sa2.out_channel,
            mlp=self.mlp3, radius_absolute=self.absolute_radius)
        self.sa4 = PointNetSetAbstraction(
            npoint=self.npoint4, radius=self.r4, nsample=self.nsample4, in_channel=3 + self.sa3.out_channel,
            mlp=self.mlp4, radius_absolute=self.absolute_radius)
        self.sa_last = PointNetSetAbstraction(
            in_channel=3 + self.sa4.out_channel, mlp=self.mlp_last, group_all=True)
        self.fc1 = nn.Linear(self.sa_last.out_channel, self.fc1_out)
        self.bn1 = nn.BatchNorm1d(self.fc1_out,
                                  momentum=0.1)
        self.drop1 = nn.Dropout(self.fc_dropout)
        self.fc2 = nn.Linear(self.fc1_out, self.fc2_out)  # Aligned with tf_pipeline -> Reduced from 576 to 256
        self.bn2 = nn.BatchNorm1d(self.fc2_out, momentum=0.1)
        self.drop2 = nn.Dropout(self.fc_dropout)
        self.fc3 = nn.Linear(self.fc2_out, num_classes)  # Aligned with tf_pipeline -> Reduced from 160 to 128

    def forward(self, data, mask=None):
        B, N, D = data.shape

        coords = data[:, :, :3]
        std_per_dim = coords.std(dim=1, unbiased=False)  # (B, 3)
        max_std_per_probe = 2 * std_per_dim.max(dim=1).values  # (B,)

        data = self.transform(data, mask)  # Feature normalization
        in_xyz, in_points = data[..., :3], data[..., 3:]
        l1_xyz, l1_points = self.sa1(in_xyz, in_points, max_std_per_probe)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points, max_std_per_probe)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points, max_std_per_probe)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points, max_std_per_probe)
        l_final_xyz, l_final_points = self.sa_last(l4_xyz, l4_points, max_std_per_probe)
        x = l_final_points.view(B, self.mlp_last[-1])
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x, l_final_points


class get_loss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(get_loss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, pred, target, trans_feat=None):
        return F.nll_loss(pred, target, weight=self.weight, reduction=self.reduction)
