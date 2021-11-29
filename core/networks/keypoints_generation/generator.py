import torch
import torch.nn.functional as F
from torch import nn

from ..building_blocks import ConvNormRelu


class AudioEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        leaky = cfg.VOICE2POSE.GENERATOR.LEAKY_RELU
        norm = cfg.VOICE2POSE.GENERATOR.NORM
        num_groups = cfg.VOICE2POSE.GENERATOR.NUM_GROUPS

        down_sample_block_1 = nn.Sequential(
            ConvNormRelu('2d', 1, 64, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('2d', 64, 64, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky),
        )
        down_sample_block_2 = nn.Sequential(
            ConvNormRelu('2d', 64, 128, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('2d', 128, 128, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky),  # downsample
        )
        down_sample_block_3 = nn.Sequential(
            ConvNormRelu('2d', 128, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('2d', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky),  # downsample
        )
        down_sample_block_4 = nn.Sequential(
            ConvNormRelu('2d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('2d', 256, 256, kernel_size=(6, 3), stride=1, padding=0, norm=norm, num_groups=num_groups, leaky=leaky),  # downsample
        )

        self.specgram_encoder_2d = nn.Sequential(
            down_sample_block_1,
            down_sample_block_2,
            down_sample_block_3,
            down_sample_block_4
        )

    def forward(self, x, num_frames):
        x = self.specgram_encoder_2d(x.unsqueeze(1))
        x = F.interpolate(x, (1, num_frames), mode='bilinear')
        x = x.squeeze(2)
        return x

class CodeMapper(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        leaky = cfg.VOICE2POSE.GENERATOR.LEAKY_RELU
        # norm = cfg.VOICE2POSE.GENERATOR.NORM
        norm = 'BN'
        num_groups = cfg.VOICE2POSE.GENERATOR.NUM_GROUPS
        code_dim = cfg.VOICE2POSE.GENERATOR.CLIP_CODE.DIMENSION
        
        self.blocks = nn.Sequential(
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),

            nn.Conv1d(256, code_dim, kernel_size=1, bias=True)
        )

    def forward(self, x):
        # noise = torch.randn(x.shape[:2], device=x.device).unsqueeze(-1).repeat(1,1,x.shape[-1])
        # x = torch.cat([x, noise], 1)

        x = torch.randn(x.shape, device=x.device)

        x = self.blocks(x)
        return x

class UNet_1D(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        leaky = cfg.VOICE2POSE.GENERATOR.LEAKY_RELU
        norm = cfg.VOICE2POSE.GENERATOR.NORM
        num_groups = cfg.VOICE2POSE.GENERATOR.NUM_GROUPS
        
        if cfg.VOICE2POSE.GENERATOR.CLIP_CODE.DIMENSION is not None:
            self.e0 = ConvNormRelu('1d', 256+cfg.VOICE2POSE.GENERATOR.CLIP_CODE.DIMENSION, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky)
        else:
            self.e0 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky)
        
        self.e1 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky)
        self.e2 = ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky)
        self.e3 = ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky)
        self.e4 = ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky)
        self.e5 = ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky)
        self.e6 = ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky)

        self.d5 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky)
        self.d4 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky)
        self.d3 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky)
        self.d2 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky)
        self.d1 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky)

    def forward(self, x):
        e0 = self.e0(x)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)

        d5 = self.d5(F.interpolate(e6, e5.size(-1), mode='linear') + e5)
        d4 = self.d4(F.interpolate(d5, e4.size(-1), mode='linear') + e4)
        d3 = self.d3(F.interpolate(d4, e3.size(-1), mode='linear') + e3)
        d2 = self.d2(F.interpolate(d3, e2.size(-1), mode='linear') + e2)
        d1 = self.d1(F.interpolate(d2, e1.size(-1), mode='linear') + e1)

        return d1

class SequenceGeneratorCNN(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.cfg = cfg

        leaky = cfg.VOICE2POSE.GENERATOR.LEAKY_RELU
        norm = cfg.VOICE2POSE.GENERATOR.NORM
        num_groups = cfg.VOICE2POSE.GENERATOR.NUM_GROUPS

        self.audio_encoder = AudioEncoder(cfg)
        self.unet = UNet_1D(cfg)
        self.decoder = nn.Sequential(
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            nn.Conv1d(256, cfg.DATASET.NUM_LANDMARKS*2, kernel_size=1, bias=True)
            )
        if cfg.VOICE2POSE.GENERATOR.CLIP_CODE.DIMENSION is not None:
            self.code_mapper = CodeMapper(cfg)

    def forward(self, x, num_frames, code=None):
        x = self.audio_encoder(x, num_frames)  # (B, C, num_frame)

        if self.cfg.VOICE2POSE.GENERATOR.CLIP_CODE.DIMENSION is not None:
            code_sequence_gen = self.code_mapper(x.detach())  # (B, C, num_frame)
            if code is None:
                code = code_sequence_gen
            else:
                code = code.unsqueeze(2).repeat([1, 1, x.shape[-1]])
            x = torch.cat([x, code], 1)
        else:
            code_sequence_gen = None

        x = self.unet(x)
        x = self.decoder(x)

        x = x.permute([0,2,1]).reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_LANDMARKS)
        return x, code_sequence_gen
