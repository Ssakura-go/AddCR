import torch
import torch.nn as nn
import unet_parts

class Model(nn.Module):
    def __init__(self, n_channels=3, bilinear=False):
        super(Model, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = unet_parts.DoubleConv(n_channels, 16)
        self.down1 = unet_parts.Down(16, 32)
        self.down2 = unet_parts.Down(32, 64)
        self.down3 = unet_parts.Down(64, 128)
        self.down4 = unet_parts.Down(128, 256)
        self.up1 = unet_parts.Up(256, 128, bilinear)
        self.up2 = unet_parts.Up(128, 64, bilinear)
        self.up3 = unet_parts.Up(64, 32, bilinear)
        self.up4 = unet_parts.Up(32, 16, bilinear)
        self.outc_img = unet_parts.OutConv(16, n_channels)
        # self.outc_line = unet_parts.OutConv(16, n_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        image = self.outc_img(x)
        # line = self.outc_line(x)
        # return image, line
        return image

    #     self.inc = unet_parts.DoubleConv(n_channels, 64)
    #     self.down1 = unet_parts.Down(64, 128)
    #     self.down2 = unet_parts.Down(128, 256)
    #     self.down3 = unet_parts.Down(256, 512)
    #     factor = 2 if bilinear else 1
    #     self.down4 = unet_parts.Down(512, 1024 // factor)
    #     self.up1 = unet_parts.Up(1024, 512 // factor, bilinear)
    #     self.up2 = unet_parts.Up(512, 256 // factor, bilinear)
    #     self.up3 = unet_parts.Up(256, 128 // factor, bilinear)
    #     self.up4 = unet_parts.Up(128, 64, bilinear)
    #     self.outc = unet_parts.OutConv(64, n_channels)
    #
    # def forward(self, x):
    #     x1 = self.inc(x)
    #     x2 = self.down1(x1)
    #     x3 = self.down2(x2)
    #     x4 = self.down3(x3)
    #     x5 = self.down4(x4)
    #     x = self.up1(x5, x4)
    #     x = self.up2(x, x3)
    #     x = self.up3(x, x2)
    #     x = self.up4(x, x1)
    #     logits = self.outc(x)
    #     return logits