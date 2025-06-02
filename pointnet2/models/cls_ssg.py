import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2.models.utils import PointNetSetAbstraction, Transform, UnitSphereNormalization, UnitCubeNormalization


class get_model(nn.Module):
    def __init__(self, num_classes, num_dimensions=3, transform=None, feats=None):
        super(get_model, self).__init__()
        self.register_buffer('num_classes', torch.tensor(num_classes))
        self.register_buffer('num_dimensions', torch.tensor(num_dimensions))
        self.transform = Transform(num_dimensions, transform, feats)
        # self.unit_sphere_normalization = UnitSphereNormalization(eps=1e-6)
        # self.unit_cube_normalization = UnitCubeNormalization(eps=1e-6)

        self.r1 = 0.05
        self.r2 = 0.12
        self.npoint1 = 50
        self.npoint2 = 30
        self.nsample1 = 10
        self.nsample2 = 20
        self.mlp1 = (64, 64, 128)
        self.mlp2 = (64, 64, 128)
        self.mlp3 = (128, 256, 512)
        self.fc1_out = 128
        self.fc2_out = 64
        self.sa_dropout = 0.2
        self.fc_dropout = 0.3

        self.sa1 = PointNetSetAbstraction(
            npoint=self.npoint1, radius=self.r1, nsample=self.nsample1,
            in_channel=self.transform.num_dimensions_transformed, mlp=self.mlp1, dropout=self.sa_dropout)
        self.sa2 = PointNetSetAbstraction(
            npoint=self.npoint2, radius=self.r2, nsample=self.nsample2,
            in_channel=3 + self.sa1.out_channel, mlp=self.mlp2, dropout=self.sa_dropout)
        self.sa3 = PointNetSetAbstraction(
            in_channel=3 + self.sa2.out_channel, mlp=self.mlp3, group_all=True, dropout=self.sa_dropout)
        self.fc1 = nn.Linear(self.sa3.out_channel, self.fc1_out)
        self.bn1 = nn.BatchNorm1d(self.fc1_out,
                                  momentum=0.1)
        self.drop1 = nn.Dropout(self.fc_dropout)
        self.fc2 = nn.Linear(self.fc1_out, self.fc2_out)  # Aligned with tf_pipeline -> Reduced from 576 to 256
        self.bn2 = nn.BatchNorm1d(self.fc2_out, momentum=0.1)
        self.drop2 = nn.Dropout(self.fc_dropout)
        self.fc3 = nn.Linear(self.fc2_out, num_classes)  # Aligned with tf_pipeline -> Reduced from 160 to 128

    def forward(self, data, mask=None):
        B, N, D = data.shape
        data = self.transform(data, mask)  # Feature normalization
        # data = self.unit_sphere_normalization(data) # Coordinate normalization
        in_xyz, in_points = data[..., :3], data[..., 3:]
        l1_xyz, l1_points = self.sa1(in_xyz, in_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, self.mlp3[-1])
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x, l3_points


class get_loss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(get_loss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, pred, target, trans_feat=None):
        return F.nll_loss(pred, target, weight=self.weight, reduction=self.reduction)
