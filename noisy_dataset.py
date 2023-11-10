import torch
from torchvision import transforms
import torchvision.transforms.functional as trf
import torch.utils.data as dutils
from typing import List, Tuple
from PIL import Image
import os
from os.path import join
import argparse
import shutil
from tqdm import tqdm


class NoisyDataset(dutils.Dataset):
    """
    Indexing gives (ground truth, rendered) Tensor pairs.
    Data augmentation: gives option to crop, flip.
    """

    def __init__(self, root_path: str, split: str, crop_size=(512, 512), flipx=True, flipy=True):
        self.root_path = join(root_path, split)
        self.crop_size = crop_size
        self.flipx = flipx
        self.flipy = flipy

        assert os.path.isdir(self.root_path)

        gt_root = join(self.root_path, "gt")
        self.gt_paths = [join(gt_root, i) for i in os.listdir(gt_root)]

        render_root = join(self.root_path, "render")

        d1 = os.listdir(render_root)[0]

        # Multiple
        if os.path.isdir(join(render_root, d1)):
            paths = []
            for d in os.listdir(render_root):
                path = join(render_root, d)
                paths.append([join(path, i) for i in os.listdir(path)])
            lengths = [len(i) for i in paths]
            assert len(set(lengths)) == 1, f"Directories in {render_root} have different file counts: {lengths}"
            self.render_paths = list(zip(*paths))
            self.n = len(paths)
        # Single
        else:
            self.render_paths = [[join(render_root, i)] for i in os.listdir(render_root)]
            self.n = 1

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, item):
        """Returns (gt, im) both of shape 3 x H x W where (H, W) = self.crop_size."""
        rp = self.render_paths[item][torch.randint(0, self.n, ())]
        gp = self.gt_paths[item]

        tot = transforms.ToTensor()
        rim = tot(Image.open(rp))
        gim = tot(Image.open(gp))

        assert rim.shape == gim.shape

        if self.crop_size is not None:
            c, h, w = rim.shape

            # Deal with images which are below the crop size by scaling them up
            if h < self.crop_size[0] or w < self.crop_size[1]:
                scale = max(self.crop_size[0] / h, self.crop_size[1] / w) + 0.1
                new_h = int(scale * h)
                new_w = int(scale * w)

                rim = trf.resize(rim, size=[new_h, new_w])
                gim = trf.resize(gim, size=[new_h, new_w])

            # Perform the same random crop on both gt and rendered image
            i, j, h, w = transforms.RandomCrop.get_params(rim, output_size=self.crop_size)
            rim = trf.crop(rim, i, j, h, w)
            gim = trf.crop(gim, i, j, h, w)

        if self.flipx and torch.randint(0, 2, ()) == 1:
            rim = trf.hflip(rim)
            gim = trf.hflip(gim)

        if self.flipy and torch.randint(0, 2, ()) == 1:
            rim = trf.vflip(rim)
            gim = trf.vflip(gim)

        return rim, gim


def construct_subset_dataset(root_path: str, out_path: str, split_every_n: int, scale_to: Tuple[int, int], iters: List[int] = None, scenes=None):
    """Create a 'subset' dataset containing only some scenes and iteration counts"""
    def path_list(scene, test_or_train, iters):
        x = "gt" if iters is None else f"render_{iters}"
        path = os.path.join(root_path, scene, x, test_or_train)
        return [os.path.join(path, i) for i in os.listdir(path)]

    def cpy(src, dst):
        im = Image.open(src).convert("RGB").resize(scale_to)
        im.save(dst)

    assert os.path.isdir(root_path)

    render_paths = []
    gt_paths = []

    # e.g. gt_paths = ["a", "b", "c"]
    #   render_paths = [["a1", "a2", "a3"], ...]

    # [a1, b1, c1]
    # [a2, b2, c2]

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
    parser.add_argument("-st", "--scale_to", type=int, nargs="+", help="Scale all images to this size", required=True)

    args = parser.parse_args()

    assert len(args.scale_to) == 2, "--scale_to argmuent must be given two integers '[width] [height]'."

    construct_subset_dataset(root_path=args.input,
                             out_path=args.output,
                             iters=args.iters,
                             scenes=args.scenes,
                             split_every_n=args.split_every_n,
                             scale_to=tuple(args.scale_to))



