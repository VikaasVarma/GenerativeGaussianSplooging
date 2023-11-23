"""
Evaluates ControlNet on the contents of a folder, writing the results to a new folder.
"""
import os
import sys

sys.path.append("ControlNet")  # workaround...
import noisy_dataset
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from ControlNet.cldm.model import create_model, load_state_dict
from ControlNet.cldm.ddim_hacked import DDIMSampler
import argparse
import util
from cnet.dataset import ControlNetDataset

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True, help="Root path for noisy dataset")
# '../models/control-sd-v1-5.ckpt'
# parser.add_argument("-m", "--model-path", type=str, required=True, help="ControlNet model path")
parser.add_argument("-b", "--batch-size", type=int, default=4, help="Batch size")
# './ControlNet/models/cldm_v15.yaml' or without ControlNet
parser.add_argument("-c", "--config", type=str, default="models/cldm_v15.yaml", help="Path to ControlNet config")
parser.add_argument("-p", "--prompt", type=str, default="best quality, extremely detailed", help="Text prompt")
parser.add_argument("-ck", "--ckpt_path", type=str, default=None, help="Checkpoint path to resume training")
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("-s", "--ddim-steps", type=int, default=50, help="DDIM Steps")
parser.add_argument("--control-strength", type=float, default=1.0, help="ControlNet control strength")
parser.add_argument("--guidance-scale", type=float, default=1.0, help="Guidance strength")
args = parser.parse_args()

n_prompt = ""  # TODO: experiment with this
eta = 0
guess_mode = False


def sample(xs):
    with torch.no_grad():
        B, C, H, W = xs.shape
        shape = (4, H // 8, W // 8)
        cond = {"c_concat": [xs], "c_crossattn": [model.get_learned_conditioning([args.prompt] * B)]}
        un_cond = {"c_concat": None if guess_mode else [xs],
                   "c_crossattn": [model.get_learned_conditioning([n_prompt] * B)]}

        strength = args.control_strength

        # To quote the ControlNet code: "Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01"
        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else (
                    [strength] * 13)

        samples, intermediates = ddim_sampler.sample(args.ddim_steps, B,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=args.guidance_scale,
                                                     unconditional_conditioning=un_cond)

        x_samples = model.decode_first_stage(samples)
        return x_samples  # B x C x H x W


# Construct dataset
transform = util.eval_transform(size=512)
noisy_ds = noisy_dataset.NoisyDataset(root_path=args.dataset, split="test", transform=transform)
dataset = ControlNetDataset(noisy_ds, prompt=args.prompt, permute=False)
dataloader = DataLoader(dataset, num_workers=0, batch_size=args.batch_size, shuffle=False)

model = create_model(args.config).cpu()
# model.load_state_dict(load_state_dict(args.ckpt_path, location='cuda'))
# model.load_from_checkpoint(args.ckpt_path)
model = model.cuda()
model.eval()
ddim_sampler = DDIMSampler(model)

out1 = os.path.join(args.out_dir, "input")
out2 = os.path.join(args.out_dir, "output")
os.makedirs(out1, exist_ok=True)
os.makedirs(out2, exist_ok=True)

to_pil = transforms.ToPILImage()

index = 0
for xs, ys in dataloader:
    xs = xs.to(util.device)
    pred = sample(xs)

    xs = ((xs + 1) / 2).clamp(0, 1)
    pred = ((pred + 1) / 2).clamp(0, 1)

    n = pred.shape[0]
    for i in range(n):
        to_pil(xs[i]).save(os.path.join(out1, f"{index}.png"))
        to_pil(pred[i]).save(os.path.join(out2, f"{index}.png"))
