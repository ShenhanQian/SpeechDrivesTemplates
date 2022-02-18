import torch
import torch.nn.functional as F
from torch import nn

from ..building_blocks import ConvNormRelu


class PoseSeqEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        leaky = cfg.POSE2POSE.AUTOENCODER.LEAKY_RELU
        norm = cfg.POSE2POSE.AUTOENCODER.NORM
        out_channels = cfg.POSE2POSE.AUTOENCODER.CODE_DIM * 2
        in_channels = cfg.DATASET.NUM_LANDMARKS*2
        
        self.blocks = nn.Sequential(
            ConvNormRelu('1d', in_channels, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, out_channels, downsample=True, norm=norm, leaky=leaky),
        )

    def forward(self, x):
        x = x.reshape(list(x.shape[:2])+[-1]).permute([0, 2, 1])

        x = self.blocks(x)
        x = F.interpolate(x, 1).squeeze(-1)
        
        mu = x[:, 0::2]
        logvar = x[:, 1::2]
        return mu, logvar

class PoseSeqDecoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        leaky = cfg.POSE2POSE.AUTOENCODER.LEAKY_RELU
        norm = cfg.POSE2POSE.AUTOENCODER.NORM
        in_channels = cfg.POSE2POSE.AUTOENCODER.CODE_DIM
        
        self.d5 = ConvNormRelu('1d', in_channels, 256, downsample=False, norm=norm, leaky=leaky)
        self.d4 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky)
        self.d3 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky)
        self.d2 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky)
        self.d1 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky)

        self.blocks = nn.Sequential(
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            nn.Conv1d(256, cfg.DATASET.NUM_LANDMARKS*2, kernel_size=1, bias=True)
        )

    def forward(self, x):
        x = F.interpolate(x.unsqueeze(-1), 2)

        x = self.d5(F.interpolate(x, x.shape[-1]*2, mode='linear'))
        x = self.d4(F.interpolate(x, x.shape[-1]*2, mode='linear'))
        x = self.d3(F.interpolate(x, x.shape[-1]*2, mode='linear'))
        x = self.d2(F.interpolate(x, x.shape[-1]*2, mode='linear'))
        x = self.d1(F.interpolate(x, x.shape[-1]*2, mode='linear'))

        x = self.blocks(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder = PoseSeqEncoder(cfg)
        self.decoder = PoseSeqDecoder(cfg)

    def forward(self, x, num_frames, mel=None, external_code=None):
        if external_code is not None:
            x = self.decoder(external_code)
            x = x.permute([0,2,1]).reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_LANDMARKS)
            return x, external_code, torch.zeros_like(external_code)
        mu, logvar = self.encoder(x)

        eps = torch.randn(logvar.shape, device=logvar.device)
        code = mu + torch.exp(0.5*logvar) * eps
        
        x = self.decoder(code)

        x = x.permute([0,2,1]).reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_LANDMARKS)
        return x, mu.squeeze(-1), logvar.squeeze(-1)
