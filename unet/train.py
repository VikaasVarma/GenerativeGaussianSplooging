import os

from torch.utils.data import DataLoader
from unet import model
import argparse
import noisy_dataset
import util
from util import device
import torch
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Root path for noisy dataset")
    parser.add_argument("-m", "--model-root-dir", type=str, required=True, help="UNet model root directory")
    parser.add_argument("--depth", type=int, default=3, help="UNet depth")
    parser.add_argument("--width", type=int, default=16, help="UNet width")
    parser.add_argument("-b", "--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("-lr", "--lr", type=float, default=1e-4,)
    args = parser.parse_args()

    transform = noisy_dataset.get_transform(size=512)
    train_ds = noisy_dataset.NoisyDataset(root_path=args.dataset, split="train", transform=transform)
    train_dl = DataLoader(train_ds, num_workers=0, batch_size=args.batch_size, shuffle=True)

    channels = [args.width * 2**i for i in range(args.depth)]
    model_path = os.path.join(args.model_root_dir, f"unet_depth={args.depth}_width={args.width}.ckpt")
    unet = model.UNet(channels=channels).to(device)
    optim = torch.optim.AdamW(unet.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    unet.load(optim, model_path)

    for i in range(args.epochs):
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

        with torch.no_grad():
            for j in range(xs.shape[0]):
                util.visualise_ims([xs[j], ys_pred[j], ys[j]],
                                   ["Render", "Pred", "GT"],
                                   size_mul=4)

        unet.losses.append(tloss / c)
        unet.save(optim, args.model_path)
