import torch
import numpy as np
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torch.utils.data as dutils
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def evaluate(model, dataset, batch_size, once=False):
    """Returns (mean PSNR, mean SSIM) on the given dataset."""
    dl = dutils.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    data_range = 1
    psnr = PeakSignalNoiseRatio(data_range=data_range, dim=(1, 2, 3), reduction="sum").to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=data_range, reduction="sum").to(device)

    psnr_total = 0
    ssim_total = 0

    count = 0

    with torch.no_grad():
        it = tqdm(dl)
        for xs, ys in it:
            xs = xs.to(device)
            ys = ys.to(device)

            ys_pred = model(xs)
            assert torch.min(ys_pred) >= 0 and torch.max(ys_pred) <= data_range

            psnr_total += psnr(ys_pred, ys)
            ssim_total += ssim(ys_pred, ys)
            count += xs.shape[0]

            it.set_description(f"PSNR = {psnr_total / count}, SSIM = {ssim_total / count}")

            if count == xs.shape[0]:
                ims = [xs[0], ys_pred[0], ys[0]]
                fig, axs = plt.subplots(3,1, figsize=(30, 60))
                for i in range(3):
                    axs[i].axis('off')
                    im = transforms.ToPILImage()(ims[i])
                    axs[i].imshow(im)
                plt.show()
                tqdm.write(f"SD psnr: {psnr(ys_pred[:1], ys[:1]).item(), ssim(ys_pred[:1], ys[:1]).item()}")
                tqdm.write(f"original psnr: {psnr(xs[:1], ys[:1]).item(), ssim(xs[:1], ys[:1]).item()}")

                if once: return

    return psnr_total / len(dataset), ssim_total / len(dataset)


def get_parser() -> argparse.ArgumentParser:
    """Shared args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Path to pre-trained Stable Diffusion checkpoint")
    parser.add_argument("-s", "--strength", type=float,
                        help="Denoising strength; 0 doesn't change the image, 1.0 gives a totally new image")
    parser.add_argument("-d", "--directory", type=str, help="Dataset directory")
    parser.add_argument("-g", "--cfg", type=float, help="Classifier-free guidance", default=0)
    parser.add_argument("-p", "--prompt", type=str, nargs="+", help="Prompt", default=None)
    parser.add_argument("-np", "--nprompt", type=str, nargs="+", help="Negative prompt", default=None)
    parser.add_argument("-n", "--nsteps", type=int, help="Total diffusion steps", default=50)
    return parser


def visualise_ims(images, captions=None, size_mul=2):
    def to_pil(im):
        if isinstance(im, np.ndarray):
            if im.dtype != np.uint8:
                # Assume floating point type
                im = np.max(np.min(im, 1), 0)
                im = (im * 255).astype(np.uint8)
            return Image.fromarray(im)
        if isinstance(im, torch.Tensor):
            if len(im.shape) == 4:
                assert im.shape[0] == 1
                im = im[0]
            if im.shape[-1] == 3:
                # Assumes (x, x, 3) means (H, W, C)
                im = im.permute((2, 0, 1))
            return transforms.ToPILImage()(im)
        if isinstance(im, Image.Image):
            return im
        raise NotImplementedError(f"Unknown image type: {type(im)}")

    images = [to_pil(i) for i in images]
    n = len(images)

    fig, axes = plt.subplots(1, n, figsize=(n*size_mul, 1*size_mul))
    for i in range(n):
        axes[i].axis('off')
        axes[i].imshow(images[i])
        if captions is not None:
            axes[i].title.set_text(captions[i])
    plt.show()


def eval_transform(size: int):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size),  # preserves aspect
        transforms.CenterCrop(size)
    ])
