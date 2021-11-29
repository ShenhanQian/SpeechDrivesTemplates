import torch
import torch.nn.functional as F
from torch import nn

from ..building_blocks import ConvNormRelu


class AudioEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        leaky = cfg.POSE2POSE.AUTOENCODER.LEAKY_RELU
        norm = cfg.POSE2POSE.AUTOENCODER.NORM
        num_groups = cfg.POSE2POSE.AUTOENCODER.NUM_GROUPS

        down_sample_block_1 = nn.Sequential(
            ConvNormRelu('2d', 1, 64, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('2d', 64, 64, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky),
            )
        down_sample_block_2 = nn.Sequential(
            ConvNormRelu('2d', 64, 128, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('2d', 128, 128, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky),
            )
        down_sample_block_3 = nn.Sequential(
            ConvNormRelu('2d', 128, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('2d', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky),
            )
        down_sample_block_4 = nn.Sequential(
            ConvNormRelu('2d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('2d', 256, 256, kernel_size=(6, 3), stride=1, padding=0, norm=norm, num_groups=num_groups, leaky=leaky),
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


class Poses_Encoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        leaky = cfg.POSE2POSE.AUTOENCODER.LEAKY_RELU
        norm = cfg.POSE2POSE.AUTOENCODER.NORM
        num_groups = cfg.POSE2POSE.AUTOENCODER.NUM_GROUPS
        out_channels = cfg.POSE2POSE.AUTOENCODER.CODE_DIM * 2
        in_channels = cfg.DATASET.NUM_LANDMARKS*2
        
        self.blocks = nn.Sequential(
            ConvNormRelu('1d', in_channels, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, out_channels, downsample=True, norm=norm, num_groups=num_groups, leaky=leaky),
        )

    def forward(self, x):
        x = x.reshape(list(x.shape[:2])+[-1]).permute([0, 2, 1])

        x = self.blocks(x)
        x = F.interpolate(x, 1).squeeze(-1)
        
        mu = x[:, 0::2]
        logvar = x[:, 1::2]
        return mu, logvar

class Poses_Decoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        leaky = cfg.POSE2POSE.AUTOENCODER.LEAKY_RELU
        norm = cfg.POSE2POSE.AUTOENCODER.NORM
        num_groups = cfg.POSE2POSE.AUTOENCODER.NUM_GROUPS
        in_channels = cfg.POSE2POSE.AUTOENCODER.CODE_DIM
        
        self.d5 = ConvNormRelu('1d', in_channels, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky)
        self.d4 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky)
        self.d3 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky)
        self.d2 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky)
        self.d1 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky)

        self.blocks = nn.Sequential(
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
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


class Poses_Decoder_With_Audio(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        leaky = cfg.POSE2POSE.AUTOENCODER.LEAKY_RELU
        norm = cfg.POSE2POSE.AUTOENCODER.NORM
        num_groups = cfg.POSE2POSE.AUTOENCODER.NUM_GROUPS
        in_channels = cfg.POSE2POSE.AUTOENCODER.CODE_DIM
        
        self.d5 = ConvNormRelu('1d', in_channels, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky)
        self.d4 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky)
        self.d3 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky)
        self.d2 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky)
        self.d1 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky)

        self.t5 = ConvNormRelu('1d', 256, 32, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky)
        self.t4 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky)
        self.t3 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky)
        self.t2 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky)
        self.t1 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky)

        self.blocks = nn.Sequential(
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, num_groups=num_groups, leaky=leaky),
            nn.Conv1d(256, cfg.DATASET.NUM_LANDMARKS*2, kernel_size=1, bias=True)
        )

    def forward(self, x, audio):
        x = F.interpolate(x.unsqueeze(-1), 2)

        x = self.d5(F.interpolate(x, x.shape[-1]*2, mode='linear') + self.t5(F.interpolate(audio, x.shape[-1]*2, mode='linear')))
        x = self.d4(F.interpolate(x, x.shape[-1]*2, mode='linear') + self.t4(F.interpolate(audio, x.shape[-1]*2, mode='linear')))
        x = self.d3(F.interpolate(x, x.shape[-1]*2, mode='linear') + self.t3(F.interpolate(audio, x.shape[-1]*2, mode='linear')))
        x = self.d2(F.interpolate(x, x.shape[-1]*2, mode='linear') + self.t2(F.interpolate(audio, x.shape[-1]*2, mode='linear')))
        x = self.d1(F.interpolate(x, x.shape[-1]*2, mode='linear') + self.t1(F.interpolate(audio, x.shape[-1]*2, mode='linear')))

        x = self.blocks(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder = Poses_Encoder(cfg)
        if self.cfg.POSE2POSE.AUTOENCODER.WITH_AUDIO:
            self.audio_encoder = AudioEncoder(cfg)
            self.decoder = Poses_Decoder_With_Audio(cfg)
        else:
            self.decoder = Poses_Decoder(cfg)

    def forward(self, x, num_frames, mel=None, external_code=None):
        if external_code is not None:
            x = self.decoder(external_code)
            x = x.permute([0,2,1]).reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_LANDMARKS)
            return x, external_code, torch.zeros_like(external_code)
        mu, logvar = self.encoder(x)

        eps = torch.randn(logvar.shape, device=logvar.device)
        code = mu + torch.exp(0.5*logvar) * eps
        
        if self.cfg.POSE2POSE.AUTOENCODER.WITH_AUDIO:
            audio = self.audio_encoder(mel, num_frames)
            x = self.decoder(code, audio)
        else:
            x = self.decoder(code)

        x = x.permute([0,2,1]).reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_LANDMARKS)
        return x, mu.squeeze(-1), logvar.squeeze(-1)
