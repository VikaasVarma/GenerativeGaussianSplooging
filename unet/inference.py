from torch.utils.data import DataLoader
from unet import model
import argparse
import noisy_dataset
import util
from util import device
import matplotlib.pyplot as plt
import numpy as np
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Root path for noisy dataset")
    parser.add_argument("-m", "--model-root-dir", type=str, required=True, help="UNet model root directory")
    parser.add_argument("--depth", type=int, default=3, help="UNet depth")
    parser.add_argument("--width", type=int, default=16, help="UNet width")
    parser.add_argument("-b", "--batch-size", type=int, default=4, help="Batch size")
    args = parser.parse_args()

    transform = util.eval_transform(size=512)
    test_ds = noisy_dataset.NoisyDataset(root_path=args.dataset, split="test", transform=transform)
    train_dl = DataLoader(test_ds, num_workers=0, batch_size=args.batch_size, shuffle=False)

    channels = [args.width * 2**i for i in range(args.depth)]
    model_name = f"unet_depth={args.depth}_width={args.width}"
    model_path = os.path.join(args.model_root_dir, model_name+".ckpt")
    unet = model.UNet(channels=channels).to(device)
    unet.load(None, model_path)
    plt.plot(np.arange(len(unet.losses))+1, unet.losses)
    plt.show()

    out_dir = f"out/{model_name}"
    util.inference_on_dataset(unet, test_ds, out_dir, batch_size=args.batch_size)
