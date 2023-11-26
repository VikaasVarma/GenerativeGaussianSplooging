"""Defines the baseline Stable Diffusion image-to-image model"""
import torch
import torch.nn as nn
from typing import List
import os


class ResBlock(nn.Module):
    """
    Simple res block for up/down sample groups; the residual block covers all but the first two layers.
    The first two layers are one conv and one activation.
    """

    def __init__(self, layers):
        super().__init__()
        self.fst = nn.Sequential(*layers[:2])
        self.snd = nn.Sequential(*layers[2:])

    def forward(self, xs):
        xs = self.fst(xs)
        return xs + self.snd(xs)


class UNet(nn.Module):

    def __init__(self, channels: List[int], convs_per_depth: int = 3, kernel_size: int = 3):
        super().__init__()

        def get_upsampler(in_ch, out_ch):
            # Exactly doubles size
            k = 2
            return nn.ConvTranspose2d(in_ch, out_ch, k, stride=2, padding=(k-2)//2)

        act = lambda: nn.ReLU()
        down_groups = []
        up_groups = []
        up_convs = []

        # [64, 128, 256, 512, 1024]

        prev_channels = 3
        for c in channels:
            group = []
            for i in range(convs_per_depth):
                conv = nn.Conv2d(prev_channels, out_channels=c, kernel_size=kernel_size, padding='same')
                group += [conv, act()]
                prev_channels = c

            down_groups.append(ResBlock(group))

        channels_rev = list(reversed(channels))[1:]
        for ind, c in enumerate(channels_rev):
            group = []
            # Note: num channels is halved by the upsampler but then doubled by concatenation
            #  so no change to prev_channels required
            for i in range(convs_per_depth):
                conv = nn.Conv2d(prev_channels, out_channels=c, kernel_size=kernel_size, padding='same')
                group += [conv, act()]
                prev_channels = c

            up_groups.append(ResBlock(group))

        for i in range(len(channels)-1, 0, -1):
            c1 = channels[i]
            c2 = channels[i-1]
            up_convs.append(get_upsampler(c1, c2))

        self.down_groups = nn.ModuleList(down_groups)
        self.up_groups = nn.ModuleList(up_groups)
        self.up_convs = nn.ModuleList(up_convs)
        self.final_conv = nn.Conv2d(in_channels=prev_channels, out_channels=3, kernel_size=1)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        self.losses = []

    def forward(self, xs):
        res = []
        for ind, group in enumerate(self.down_groups):
            xs = group(xs)
            if ind != len(self.down_groups)-1:
                res.append(xs)
                xs = self.downsample(xs)

        for ind, group in enumerate(self.up_groups):
            xs = self.up_convs[ind](xs)
            ys = res.pop(-1)
            assert xs.shape == ys.shape
            # Concatenate along the channels dimension
            xs = torch.cat((xs, ys), dim=1)

            xs = self.up_groups[ind](xs)

        xs = self.final_conv(xs)
        xs = nn.functional.sigmoid(xs)

        return xs

    def save(self, optim, path):
        d = {
            "params": self.state_dict(),
            "opt": optim.state_dict(),
            "losses": self.losses
        }
        torch.save(d, path)
        print("Saved to", path)

    def load(self, optim, path):
        if os.path.isfile(path):
            d = torch.load(path)
            self.load_state_dict(d["params"])
            self.losses = d["losses"]
            if optim is not None:
                optim.load_state_dict(d["opt"])
        else:
            print("No checkpoint found at", path)
