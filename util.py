import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torch.utils.data as dutils
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def evaluate(model, dataset, batch_size):
    """Returns (mean PSNR, mean SSIM) on the given dataset."""
    dl = dutils.DataLoader(dataset, batch_size=batch_size, shuffle=True)
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

            # ims = [xs[0], ys[0], ys_pred[0]]
            # fig, axs = plt.subplots(3,1, figsize=(15, 30))
            # for i in range(3):
            #     axs[i].axis('off')
            #     im = transforms.ToPILImage()(ims[i])
            #     axs[i].imshow(im)
            # plt.show()

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
