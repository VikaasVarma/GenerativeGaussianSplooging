"""Defines the baseline Stable Diffusion image-to-image model"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List
import argparse
from tqdm import tqdm
import os
import pytorch_lightning as pl

from util import device
import util
import noisy_dataset


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
            "opt": optim.state_dict()
        }
        torch.save(d, path)
        print("Saved to", path)

    def load(self, optim, path):
        if os.path.isfile(path):
            d = torch.load(path)
            self.load_state_dict(d["params"])
            optim.load_state_dict(d["optim"])
        else:
            print("No checkpoint found at", path)


# TODO: Move to submodule, separate unet/train_unet/eval_unet files
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Root path for noisy dataset")
    parser.add_argument("-m", "--model-path", type=str, required=True, help="ControlNet model path")
    parser.add_argument("-b", "--batch-size", type=int, default=4, help="Batch size")
    args = parser.parse_args()

    transform = noisy_dataset.get_transform(size=512)
    train_ds = noisy_dataset.NoisyDataset(root_path=args.dataset, split="train", transform=transform)
    train_dl = DataLoader(train_ds, num_workers=0, batch_size=args.batch_size, shuffle=True)

    # 1) make model
    unet = UNet(channels=[16, 32, 64]).to(device)
    optim = torch.optim.AdamW(unet.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    unet.load(optim, args.model_path)

    # 2) train
    for i in range(10):
        tloss = 0
        c = 0
        it = tqdm(train_dl)
        for xs, ys in it:
            xs = xs.to(device)
            ys = ys.to(device)
            optim.zero_grad()
            ys_pred = unet(xs)
            loss = loss_fn(ys_pred, ys)
            loss.backward()
            optim.step()
            tloss += loss.item()
            c += 1
            it.set_description(f"Loss = {tloss / c:.4f}")
        print(f"AVERAGE LOSS FOR ROUND {i+1}:", tloss / c)

        unet.save(optim, args.model_path)

        with torch.no_grad():
            for j in range(xs.shape[0]):
                util.visualise_ims([xs[j], ys_pred[j], ys[j]],
                                   ["Render", "Pred", "GT"],
                                   size_mul=4)

    # 3) eval
    # psnr, ssim = util.evaluate(model, test_ds, batch_size=1)


