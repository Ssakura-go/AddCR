import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction):
        super(ChannelAttention, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.module(x)


class RCAB(nn.Module):
    def __init__(self, num_features, reduction):
        super(RCAB, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction)
        )

    def forward(self, x):
        return x + self.module(x)


class RG(nn.Module):
    def __init__(self, num_features, num_rcab, reduction):
        super(RG, self).__init__()
        self.module = [RCAB(num_features, reduction) for _ in range(num_rcab)]
        self.module.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return x + self.module(x)


class RCAN(nn.Module):
    def __init__(self, args):
        super(RCAN, self).__init__()
        scale = args.scale
        num_features = args.num_features
        num_rg = args.num_rg
        num_rcab = args.num_rcab
        reduction = args.reduction

        self.sf = nn.Conv2d(3, num_features, kernel_size=3, padding=1)
        self.rgs = nn.Sequential(*[RG(num_features, num_rcab, reduction) for _ in range(num_rg)])
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        # self.upscale_image = nn.Sequential(
        #     nn.Conv2d(num_features, num_features * (scale ** 2), kernel_size=3, padding=1),
        #     nn.PixelShuffle(scale)
        # )
        # self.upscale_line = nn.Sequential(
        #     nn.Conv2d(num_features, num_features * (scale ** 2), kernel_size=3, padding=1),
        #     nn.PixelShuffle(scale)
        # )
        self.conv_image = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        # self.conv_s = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv2_image = nn.Conv2d(num_features, 3, kernel_size=3, padding=1)
        # self.conv2_s = nn.Conv2d(num_features, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.sf(x)
        residual = x
        x = self.rgs(x)
        x = self.conv1(x)
        # print("x1:{}".format(x.shape))
        x += residual
        # x_image = self.upscale_image(x)
        # x_s = self.upscale_line(x)
        # print("x2:{}".format(x.shape))
        x_image = self.conv_image(x)
        # x_s = self.conv_s(x)
        x_image = self.conv2_image(x_image)
        # x_s = self.conv2_s(x_s)
        return x_image