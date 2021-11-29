import torch
import torch.nn.functional as F
from torch import nn

from ..building_blocks import ConvNormRelu, FCNormRelu


class PoseSequenceDiscriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        leaky = self.cfg.VOICE2POSE.POSE_DISCRIMINATOR.LEAKY_RELU

        self.seq = nn.Sequential(
            ConvNormRelu('1d', cfg.DATASET.NUM_LANDMARKS*2, 256, downsample=True, leaky=leaky),  # B, 256, 64
            ConvNormRelu('1d', 256, 512, downsample=True, leaky=leaky),  # B, 512, 32
            ConvNormRelu('1d', 512, 1024, kernel_size=3, stride=1, padding=1, leaky=leaky),  # B, 1024, 16
            nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1, bias=True)  # B, 1, 16
        )

    def forward(self, x):
        x = x.reshape(x.size(0), x.size(1), -1).transpose(1, 2)
        x = self.seq(x)
        x = x.squeeze(1)
        return x

class CodeDiscriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        leaky = self.cfg.VOICE2POSE.POSE_DISCRIMINATOR.LEAKY_RELU
        # norm = cfg.VOICE2POSE.GENERATOR.NORM
        norm = 'BN'

        self.seq = nn.Sequential(
            FCNormRelu(cfg.VOICE2POSE.GENERATOR.CLIP_CODE.DIMENSION, 256, norm=norm, leaky=leaky),  # B, 256
            FCNormRelu(256, 256, norm=norm, leaky=leaky),
            FCNormRelu(256, 256, norm=norm, leaky=leaky),

            FCNormRelu(256, 256, norm=norm, leaky=leaky),
            FCNormRelu(256, 256, norm=norm, leaky=leaky),
            FCNormRelu(256, 256, norm=norm, leaky=leaky),
            FCNormRelu(256, 256, norm=norm, leaky=leaky),

            nn.Linear(256, 1, bias=True)
        )
    
    def forward(self, x):
        x = self.seq(x)
        return x
