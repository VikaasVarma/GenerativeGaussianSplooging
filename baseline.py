"""Defines the baseline Stable Diffusion image-to-image model. If used as a command, applies"""
import torch
import torch.nn as nn
import diffusers
import argparse
from util import device, evaluate
from noisy_dataset import NoisyDataset


class ImageToImageBaseline(nn.Module):
    def __init__(self, sd_pipeline: diffusers.StableDiffusionImg2ImgPipeline, strength: float, nsteps: int = 50,
                 prompt=None, negative_prompt=None, cfg: float = 0):
        super().__init__()
        self.sd = sd_pipeline
        self.strength = strength
        self.nsteps = nsteps

        if prompt is not None or negative_prompt is not None:
            assert cfg > 0
        else:
            assert cfg == 0  # this leads to unconditional sampling
        self.cfg = cfg
        self.prompt = prompt if prompt is not None else ""
        self.negative_prompt = negative_prompt

    def forward(self, xs):
        # xs : B x C x H x W
        (out, ), _ = self.sd(
            prompt=self.prompt,
            image=xs,
            strength=self.strength,
            num_inference_steps=self.nsteps,
            guidance_scale=self.cfg,
            negative_prompt=self.negative_prompt,
            return_dict=False,
            output_type="pt"  # return a Tensor
        )
        # Given batch size 1, the pipeline automatically removes this dimension
        if len(out.shape) == 3:
            return out.unsqueeze(0)
        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Path to pre-trained Stable Diffusion checkpoint")
    parser.add_argument("-s", "--strength", type=float,
                        help="Denoising strength; 0 doesn't change the image, 1.0 gives a totally new image")
    parser.add_argument("-d", "--directory", type=str, help="Dataset directory")
    parser.add_argument("-g", "--cfg", type=float, help="Classifier-free guidance", default=0)
    parser.add_argument("-p", "--prompt", type=str, nargs="+", help="Prompt", default=None)
    parser.add_argument("-np", "--nprompt", type=str, nargs="+", help="Negative prompt", default=None)

    args = parser.parse_args()
    prompt = " ".join(args.prompt) if args.prompt is not None else None
    nprompt = " ".join(args.nprompt) if args.nprompt is not None else None

    print(f"Running with strength={args.strength}, cfg={args.cfg}, prompt={prompt}, nprompt={nprompt}")

    test_ds = NoisyDataset(root_path=args.directory, split="test", transform=None)

    sd = diffusers.StableDiffusionImg2ImgPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    # sd.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
    sd.set_progress_bar_config(disable=True)
    assert sd.safety_checker is None
    model = ImageToImageBaseline(sd, strength=args.strength, prompt=prompt, negative_prompt=nprompt,
                                 cfg=args.cfg).to(device)

    psnr, ssim = evaluate(model, test_ds, batch_size=1)
    print(f"Results with strength={args.strength}, cfg={args.cfg}, prompt={prompt}, nprompt={nprompt}")
    print(f"Im2im: PSNR={psnr.item():.3f}, SSIM={ssim.item():.4f}")

    psnr, ssim = evaluate(lambda x: x, test_ds, batch_size=8)
    print(f"Dataset: PSNR={psnr.item():.3f}, SSIM={ssim.item():.4f}")


