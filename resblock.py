import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(Block, self).__init__()
        self.same_shape = same_shape
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
        )
        if not same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.block(x)
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(out + x)

class Block2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Block2, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        out = self.block(x)
        return F.relu(out + x)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = self._make_layer(3, 16)
        self.layer2 = self._make_layer(16, 16)
        self.layer3_class = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.layer3_line = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.layer4_class = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.layer4_line = nn.Conv2d(16, 1,  kernel_size=3, padding=1)

    def _make_layer(self, in_channel, out_channel):
        layers = []
        layers.append(Block(in_channel, out_channel, same_shape=False))
        layers.append(Block(out_channel, out_channel, same_shape=True))
        return nn.Sequential(*layers)

    def _make_layer2(self, in_channel, out_channel):
        layers = []
        layers.append(Block2(in_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x_class = self.layer3_class(x)
        x_line = self.layer3_line(x)
        x_class = self.layer4_class(x_class)
        x_line = self.layer4_line(x_line)

        return x_class, x_line