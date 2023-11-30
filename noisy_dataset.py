import torch
from torchvision import transforms
import torchvision.transforms.functional as trf
import numpy as np
import torch.utils.data as dutils
from typing import List, Tuple
from PIL import Image
import os
from os.path import join
import argparse
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

import util


# Flips, crops
def get_transform(size=512, crop_size=0.3):
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=size, scale=(crop_size, 1.0)),
    ])
    return t


class NoisyDataset(dutils.Dataset):
    """
    Indexing gives (ground truth, rendered) Tensor pairs.
    Data augmentation: gives option to crop, flip.
    """

    def __init__(self, root_path: str, split: str, return_all=False, transform=None, load_into_memory=False):
        self.root_path = join(root_path, split)
        self.return_all = return_all
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.split = split

        assert os.path.isdir(self.root_path)

        gt_root = join(self.root_path, "gt")
        items = sorted(os.listdir(gt_root))
        self.gt_paths = [join(gt_root, i) for i in items]

        render_root = join(self.root_path, "render")

        d1 = os.listdir(render_root)[0]

        # Multiple
        if os.path.isdir(join(render_root, d1)):
            paths = []
            for d in os.listdir(render_root):
                path = join(render_root, d)
                paths.append([join(path, i) for i in items])
            self.render_paths = list(zip(*paths))
            self.n = len(paths)
        # Single
        else:
            self.render_paths = [[join(render_root, i)] for i in items]
            self.n = 1

        self.load_into_memory = load_into_memory

        def save_im(path):
            with Image.open(path) as im:
                return np.array(im)

        if load_into_memory:
            print("Loading entire dataset into RAM...")
            self.gt_ims = [save_im(p) for p in self.gt_paths]
            self.render_ims = []
            for x in self.render_paths:
                if isinstance(x, list):
                    self.render_ims.append([save_im(p) for p in x])
                else:
                    self.render_ims.append([save_im(x)])

    def __len__(self):
        return len(self.gt_paths)

    def _load_gt(self, index):
        if self.load_into_memory:
            return Image.fromarray(self.gt_ims[index])
        return Image.open(self.gt_paths[index])

    def _load_render(self, index, index2):
        if self.load_into_memory:
            return Image.fromarray(self.render_ims[index][index2])
        return Image.open(self.render_paths[index][index2])

    def __getitem__(self, item):
        """Returns (gt, rendered_im) both of shape 3 x H x W where (H, W) = self.crop_size."""
        state = torch.get_rng_state()
        gim = self.transform(self._load_gt(item))

        if self.return_all:
            rims = []
            for i in range(len(self.render_paths)):
                torch.set_rng_state(state)
                rim = self.transform(self._load_render(item, i))
                rims.append(rim)
            return tuple(rims), gim
        else:
            ind = item % self.n if self.split == "test" else torch.randint(0, self.n, ())

            torch.set_rng_state(state)
            rim = self.transform(self._load_render(item, ind))

            return rim, gim

    def visualise(self, item):
        render, gt = self[item]
        util.visualise_ims([render, gt], captions=["Render", "Ground Truth"])


def construct_subset_dataset(root_path: str, out_path: str, split_every_n: int, scale_to: Tuple[int, int],
                             iters: List[int] = None, scenes=None):
    """Create a 'subset' dataset containing only some scenes and iteration counts"""

    def path_list(scene, test_or_train, iters):
        x = "gt" if iters is None else f"render_{iters}"
        path = os.path.join(root_path, scene, x, test_or_train)
        return [os.path.join(path, i) for i in os.listdir(path)]

    def cpy(src, dst):
        im = Image.open(src).convert("RGB")
        if scale_to is not None:
            im = im.resize(scale_to)
        im.save(dst)

    assert os.path.isdir(root_path)

    render_paths = []
    gt_paths = []

    # e.g. gt_paths = ["a", "b", "c"]
    #   render_paths = [["a1", "a2", "a3"], ...]

    for scene in os.listdir(root_path):
        # If scenes is specified, skip scene if not in the list
        if scenes is not None and scene not in scenes:
            continue
        gt_paths += path_list(scene, "train", None)
        gt_paths += path_list(scene, "test", None)

        rpaths = []
        for i in os.listdir(os.path.join(root_path, scene)):
            if i.startswith("render_"):
                it = int(i[7:])
                if iters is None or it in iters:
                    rpaths.append(path_list(scene, "train", it) + path_list(scene, "test", it))

        render_paths += list(zip(*rpaths))

    out_gt = [os.path.join(out_path, "train", "gt"), os.path.join(out_path, "test", "gt")]
    out_rn = [os.path.join(out_path, "train", "render"), os.path.join(out_path, "test", "render")]

    os.makedirs(out_gt[0], exist_ok=True)
    os.makedirs(out_gt[1], exist_ok=True)
    os.makedirs(out_rn[0], exist_ok=True)
    os.makedirs(out_rn[1], exist_ok=True)
    n = len(render_paths[0])
    if n > 1:
        for i in range(n):
            os.makedirs(os.path.join(out_rn[0], str(i)), exist_ok=True)
            os.makedirs(os.path.join(out_rn[1], str(i)), exist_ok=True)

    for i, path in tqdm(enumerate(gt_paths)):
        ind = 1 if i % split_every_n == 0 else 0
        path2 = os.path.join(out_gt[ind], f"{i}.png")
        # shutil.copy(path, path2)
        cpy(path, path2)

    for i, paths in tqdm(enumerate(render_paths)):
        ind = 1 if i % split_every_n == 0 else 0
        for j, path in enumerate(paths):
            p = f"{j}/{i}.png" if n > 1 else f"{i}.png"
            path2 = os.path.join(out_rn[ind], p)
            # shutil.copy(path, path2)
            cpy(path, path2)


if __name__ == "__main__":
    # Used as a script, this file converts a dataset into a test/train split
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Dataset root")
    parser.add_argument("-o", "--output", type=str, help="Location to store dataset subset")
    parser.add_argument("-is", "--iters", type=int, nargs="+", help="List of iters to include", default=None)
    parser.add_argument("-s", "--scenes", type=str, nargs="+", help="List of scenes to include", default=None)
    parser.add_argument("-n", "--split_every_n", type=int, help="Every nth image is added to test", default=10)
    parser.add_argument("-st", "--scale_to", type=int, nargs="+", help="Scale all images to this size", default=None)

    args = parser.parse_args()

    assert args.scale_to is None or len(args.scale_to) == 2, "--scale_to argmuent must be given two integers '[width] [height]'."

    construct_subset_dataset(root_path=args.input,
                             out_path=args.output,
                             iters=args.iters,
                             scenes=args.scenes,
                             split_every_n=args.split_every_n,
                             scale_to=tuple(args.scale_to) if args.scale_to is not None else None)
