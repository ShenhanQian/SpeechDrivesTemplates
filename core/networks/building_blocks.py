from torch import nn


class ConvNormRelu(nn.Module):
    def __init__(self, conv_type='1d', in_channels=3, out_channels=64, downsample=False,
                 kernel_size=None, stride=None, padding=None, norm='BN', leaky=False):
        super().__init__()
        if kernel_size is None:
            if downsample:
                kernel_size, stride, padding = 4, 2, 1
            else:
                kernel_size, stride, padding = 3, 1, 1

        if conv_type == '2d':
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            )
            if norm == 'BN':
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm == 'IN':
                self.norm = nn.InstanceNorm2d(out_channels)
            else:
                raise NotImplementedError
        elif conv_type == '1d':
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            )
            if norm == 'BN':
                self.norm = nn.BatchNorm1d(out_channels)
            elif norm == 'IN':
                self.norm = nn.InstanceNorm1d(out_channels)
            else:
                raise NotImplementedError
        nn.init.kaiming_normal_(self.conv.weight)

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True) if leaky else nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if isinstance(self.norm, nn.InstanceNorm1d):
            x = self.norm(x.permute((0, 2, 1))).permute((0, 2, 1))  # normalize on [C]
        else:
            x = self.norm(x)
        x = self.act(x)
        return x

class FCNormRelu(nn.Module):
    def __init__(self, in_features=256, out_features=256, norm='BN', leaky=False):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        if norm == 'BN':
                self.norm = nn.BatchNorm1d(out_features)
        elif norm == 'IN':
            self.norm = nn.InstanceNorm1d(out_features)
        nn.init.kaiming_normal_(self.fc.weight)

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True) if leaky else nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        if isinstance(self.norm, nn.InstanceNorm1d):
            x = self.norm(x.unsqueeze(-1)).squeeze(-1)
        else:
            x = self.norm(x)
        x = self.act(x)
        return x