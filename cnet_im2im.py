"""
Evaluates ControlNet on a directory, writing the results to an output directory.
Used as a tool for improving Gaussian splatting scenes.
"""
import diffusers
import torch
from torchvision import transforms
import os
from os.path import join
from PIL import Image
from tqdm import tqdm

from baseline import ImageToImageBaseline
import util
from util import device

if __name__ == "__main__":
    parser = util.get_parser()
    parser.add_argument("--controlnet", type=str, required=True, help="Path to ControlNet model")
    parser.add_argument("--control-strength", type=float, default=1)
    parser.add_argument("-o", "--out", type=str, help="Output directory")
    parser.add_argument("-b", "--batch-size", type=int)
    args = parser.parse_args()
    prompt = " ".join(args.prompt) if args.prompt is not None else None
    nprompt = " ".join(args.nprompt) if args.nprompt is not None else None

    print("Writing to", args.out)
    os.makedirs(args.out, exist_ok=True)

    controlnet = diffusers.ControlNetModel.from_single_file(args.controlnet, torch_dtype=torch.float16).to(util.device)
    sd = diffusers.StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.model,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    sd.set_progress_bar_config(disable=True)
    sd.enable_xformers_memory_efficient_attention()
    model = ImageToImageBaseline(sd,
                                 strength=args.strength,
                                 prompt=prompt,
                                 negative_prompt=nprompt,
                                 cfg=args.cfg,
                                 nsteps=args.nsteps).to(device)
    # guidance_strength=args.guidance_strength

    transform = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    names = os.listdir(args.directory)

    for start in tqdm(range(0, len(names), args.batch_size), desc="Applying ControlNet"):
        end = min(start + args.batch_size, len(names))

        batch = []

        for i in range(start, end):
            im = Image.open(join(args.directory, names[i]))
            batch.append(transform(im))

        inp = torch.stack(batch).to(device)
        out = model(inp, control_image=2*inp-1, controlnet_conditioning_scale=args.control_strength)

        for i in range(start, end):
            to_pil(out[i - start]).save(join(args.out, names[i]))
