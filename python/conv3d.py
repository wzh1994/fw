import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3d(num_args=449):
    return DistModule(num_args)


class DistModule(nn.Module):
    def __init__(self, num_args=449):
        super(DistModule, self).__init__()
        self.module = Conv3D(num_args)
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)


class Conv3D(nn.Module):

    def __init__(self, num_args=449):
        super(Conv3D, self).__init__()
        self.conv1 = BasicConv3d(6, 24, kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.unit1 = BaseUnit(24, 32)
        self.unit2 = BaseUnit(32, 48)
        self.unit3 = BaseUnit(48, 72)
        self.unit4 = BaseUnit(72, 96)
        self.unit5 = BaseUnit(96, 192)
        self.unit6 = BaseUnit(192, 256)
        self.fc1 = nn.Linear(3072, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, num_args)

    def forward(self, x):
        # 49*333*333*6
        x = self.conv1(x)
        # 49*166*166*24
        x = self.unit1(x)
        # 49*166*166*32
        x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # 49*84*84*32
        x = self.unit2(x)
        # 49*84*84*48
        x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # 49*42*42*48
        x = self.unit3(x)
        # 49*42*42*72
        x = F.max_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # 24*21*21*72
        x = self.unit4(x)
        # 24*21*21*96
        x = F.max_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # 12*10*10*96
        x = self.unit5(x)
        # 12*10*10*192
        x = F.max_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # 6*5*5*192
        x = self.unit6(x)
        # 6*5*5*256
        x = F.max_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # 3*2*2*256
        x = x.view(x.size(0), -1)
        # 3072
        x = self.fc1(x)
        x = torch.tanh(x)
        x = F.dropout(x, training=self.training)
        # 2048
        x = self.fc2(x)
        x = torch.tanh(x)
        x = F.dropout(x, training=self.training)
        # 2048
        x = self.fc3(x)
        # num_args
        return x


class BaseUnit(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(BaseUnit, self).__init__()
        out_channel = int(out_channels / 8)
        self.branch1_1x1x1 = BasicConv3d(in_channels, out_channel, kernel_size=1)

        self.branch2_1x1x1 = BasicConv3d(in_channels, out_channel, kernel_size=1)
        self.branch2_1x1x3 = BasicConv3d(out_channel, out_channel, kernel_size=(1, 1, 3), padding=(0, 0, 1))
        self.branch2_1x3x1 = BasicConv3d(out_channel, out_channel, kernel_size=(1, 3, 1), padding=(0, 1, 0))
        self.branch2_3x1x1 = BasicConv3d(out_channel, out_channel, kernel_size=(3, 1, 1), padding=(1, 0, 0))

        self.branch3_1x1x1 = BasicConv3d(in_channels, out_channel, kernel_size=1)
        self.branch3_3x3x3 = BasicConv3d(out_channel, out_channel, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.branch3_1x1x3 = BasicConv3d(out_channel, out_channel, kernel_size=(1, 1, 3), padding=(0, 0, 1))
        self.branch3_1x3x1 = BasicConv3d(out_channel, out_channel, kernel_size=(1, 3, 1), padding=(0, 1, 0))
        self.branch3_3x1x1 = BasicConv3d(out_channel, out_channel, kernel_size=(3, 1, 1), padding=(1, 0, 0))

        self.branch4_1x1x1 = BasicConv3d(in_channels, out_channel, kernel_size=1)

    def forward(self, x):
        branch1x1x1 = self.branch1_1x1x1(x)

        branch3x3x3 = self.branch2_1x1x1(x)
        branch3x3x3_1 = self.branch2_1x1x3(branch3x3x3)
        branch3x3x3_2 = self.branch2_1x3x1(branch3x3x3)
        branch3x3x3_3 = self.branch2_3x1x1(branch3x3x3)

        branch5x5x5 = self.branch3_1x1x1(x)
        branch5x5x5 = self.branch3_3x3x3(branch5x5x5)
        branch5x5x5_1 = self.branch3_1x1x3(branch5x5x5)
        branch5x5x5_2 = self.branch3_1x3x1(branch5x5x5)
        branch5x5x5_3 = self.branch3_3x1x1(branch5x5x5)

        branch_pool = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch4_1x1x1(branch_pool)

        outputs = [branch1x1x1, branch3x3x3_1, branch3x3x3_2, branch3x3x3_3,
                   branch5x5x5_1, branch5x5x5_2, branch5x5x5_3, branch_pool]

        # merge channel
        return torch.cat(outputs, 1)


class BasicConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
