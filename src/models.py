import torch
import torch.nn as nn


#RCAN
class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(num_features, num_features // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

class RCAB(nn.Module):
    def __init__(self, channels, reduction=16):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            ChannelAttention(channels, reduction)
        )

    def forward(self, x):
        res = self.body(x)
        return res + x

class RCAN(nn.Module):
    def __init__(self, num_channels, num_res_blocks=16, n_feats=64):
        super(RCAN, self).__init__()

        self.head = nn.Conv2d(num_channels, n_feats, kernel_size=3, padding=1)

        self.body = nn.Sequential(*[
            RCAB(n_feats) for _ in range(num_res_blocks)
        ])

        self.body_end = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)

        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
        )

        self.tail = nn.Conv2d(n_feats, num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x_head = self.head(x)

        res = self.body(x_head)
        res = self.body_end(res)
        res += x_head

        x_up = self.upsample(res)
        out = self.tail(x_up)

        return torch.clamp(out, 0.0, 1.0)

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss

#ResSR

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv2(self.relu(self.conv1(x)))
        return x + residual

class ResSR(nn.Module):
    def __init__(self, num_channels, num_res_blocks=8, n_feats=64):
        super(ResSR, self).__init__()

        self.head = nn.Conv2d(num_channels, n_feats, kernel_size=3, padding=1)

        self.body = nn.Sequential(*[
            ResidualBlock(n_feats) for _ in range(num_res_blocks)
        ])

        self.body_end = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)

        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

        self.tail = nn.Conv2d(n_feats, num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x_head = self.head(x)

        res = self.body(x_head)
        res = self.body_end(res)
        res += x_head

        x_up = self.upsample(res)
        out = self.tail(x_up)

        return torch.clamp(out, 0.0, 1.0)

#FSRCNN

class FSRCNN_Y(nn.Module):
    def __init__(self):
        scale_factor = 2
        d, s, m = 56, 12, 4
        channels = 1
        super(FSRCNN_Y, self).__init__()

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(channels, d, kernel_size=5, padding=2), nn.PReLU(d))
        self.shrink = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s))

        map_layers = []
        for _ in range(m):
            map_layers.extend([nn.Conv2d(s, s, kernel_size=3, padding=1), nn.PReLU(s)])
        self.mapping = nn.Sequential(*map_layers)

        self.expand = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d))
        self.deconv = nn.ConvTranspose2d(d, channels, kernel_size=9, stride=scale_factor, padding=4, output_padding=scale_factor - 1)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.shrink(x)
        x = self.mapping(x)
        x = self.expand(x)
        x = self.deconv(x)
        return torch.clamp(x, 0.0, 1.0)
