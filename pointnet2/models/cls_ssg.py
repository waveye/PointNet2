import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2.models.utils import PointNetSetAbstraction, Transform


class get_model(nn.Module):
    def __init__(self, num_classes, num_dimensions=3, transform=None):
        super(get_model, self).__init__()
        self.register_buffer('num_classes', torch.tensor(num_classes))
        self.register_buffer('num_dimensions', torch.tensor(num_dimensions))
        self.transform = Transform(num_dimensions, transform)

        self.sa1 = PointNetSetAbstraction(
            npoint=54, radius=0.2, nsample=28,
            in_channel=self.transform.num_dimensions_transformed, mlp=(64, 64, 128))
        self.sa2 = PointNetSetAbstraction(
            npoint=22, radius=0.2, nsample=8,
            in_channel=3 + self.sa1.out_channel, mlp=(64, 64, 128))
        self.sa3 = PointNetSetAbstraction(
            in_channel=3 + self.sa2.out_channel, mlp=(256, 512, 1024), group_all=True)
        self.fc1 = nn.Linear(self.sa3.out_channel, 576)
        self.bn1 = nn.BatchNorm1d(576)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(576, 160)
        self.bn2 = nn.BatchNorm1d(160)
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(160, num_classes)

    def forward(self, data, mask=None):
        B, N, D = data.shape
        data = self.transform(data, mask)
        in_xyz, in_points = data[..., :3], data[..., 3:]
        l1_xyz, l1_points = self.sa1(in_xyz, in_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
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
