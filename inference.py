import argparse
import os

from PIL import Image
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from UniControl.cldm.model import create_model, load_state_dict
from UniControl.cldm.ddim_unicontrol_hacked import DDIMSampler
from noisy_dataset import NoisyDataset

# This line fails if placed earlier
import numpy as np

parser = argparse.ArgumentParser(description="args")
parser.add_argument(
    "--ckpt",
    type=str,
    default="./UniControl/ckpts/unicontrol_v1.1.ckpt",
    help="$path to checkpoint",
)
parser.add_argument("--input_dir", type=str, default="../inputs", help="input path")
parser.add_argument("--output_dir", type=str, default="../outputs", help="output path")
parser.add_argument("--seed", default=-1, help="seed")
parser.add_argument("--steps", default=50, help="DDIM Steps")
parser.add_argument("--config", default="./UniControl/models/cldm_v15_unicontrol.yaml", help="option of config")
parser.add_argument("--prompt", default='best quality, extremely detailed', help="text Prompt")
parser.add_argument("--batch_size", default=1, help="Batch Size")

args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(args.input_dir):
    os.makedirs(args.input_dir)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


# Load Model
model = create_model(args.config).cpu()
model.load_state_dict(load_state_dict(args.ckpt, location=device), strict=False)
model.eval()

ddim_sampler = DDIMSampler(model)
dataloader = DataLoader(NoisyDataset(args.input_dir, split="test"), batch_size=args.batch_size, shuffle=False)

with torch.no_grad():
    for idx, batch in enumerate(dataloader):
        seed = np.random.randint(0, 1 << 16) if args.seed == -1 else args.seed
        pl.seed_everything(seed)

        # Ensure batch shape is B X C X H X W
        task_desc = {"name": "control_blur", "instruction": "deblur image to clean image", "feature": model.get_learned_conditioning("deblur image to clean image")[:,:1,:]}
        conditioning = {"c_concat" : [batch], "c_crossattn" : [model.get_learned_conditioning([args.prompt] * args.batch_size)], "task": task_desc}

        samples, intermediates = ddim_sampler.sample(args.steps, args.batch_size, batch.shape[1:], conditioning, verbose=False, eta=0, unconditional_guidance_scale=0)

        images = model.decode_first_stage(samples)
        images = torch.clamp((images + 1) / 2, 0, 1)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        
        for image in images:
            image = Image.fromarray(image.astype(np.uint8))
            image.save(os.path.join(args.output_dir, f"{idx:05}.png"))