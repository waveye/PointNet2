import torch.nn as nn
import torch.nn.functional as F

from .pointnet2_utils import PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self, num_class, num_dimensions=3):
        super(get_model, self).__init__()
        self.num_dimensions = num_dimensions
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32, in_channel=num_dimensions, mlp=(64, 64, 128))
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64, in_channel=3 + self.sa1.out_channel, mlp=(128, 128, 256))
        self.sa3 = PointNetSetAbstraction(
            in_channel=3 + self.sa2.out_channel, mlp=(256, 512, 1024), group_all=True)
        self.fc1 = nn.Linear(self.sa3.out_channel, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, data):
        B, N, D = data.shape
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
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
