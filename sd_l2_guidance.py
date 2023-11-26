"""An image-to-image Stable Diffusion model guided by the L2 distance to the original latent"""
import torch
import torch.nn as nn
import diffusers
import torchvision.transforms

from util import device
import util
from noisy_dataset import NoisyDataset


class GuidedImageToImage(nn.Module):
    def __init__(self, sd_pipeline: diffusers.StableDiffusionImg2ImgPipeline, strength: float, nsteps: int = 50,
                 prompt=None, negative_prompt=None, cfg: float = 0, guidance_strength: float = 100):
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

        self.guidance_strength = guidance_strength

    def forward(self, xs):
        with torch.no_grad():
            xs_p = self.sd.image_processor.preprocess(xs)
            xs_p = xs_p.half()
            goal_latents = self.sd.vae.encode(xs_p).latent_dist.sample()
            goal_latents = goal_latents * self.sd.vae.config.scaling_factor

        def callback(pipe, step_index, timestep, kwargs):
            latents = kwargs["latents"]
            with torch.enable_grad():
                latents = torch.nn.Parameter(latents)
                loss = (latents - goal_latents).pow(2).mean()
                loss.backward()
                g = latents.grad

            den = torch.sqrt(1 - pipe.scheduler.alphas_cumprod[step_index])
            num = 1 - pipe.scheduler.alphas[step_index]
            mul = 1 / torch.sqrt(pipe.scheduler.alphas[step_index])
            s = self.guidance_strength * mul * num / den

            latents -= s * g
            kwargs["latents"] = latents.data
            return kwargs

        # xs : B x C x H x W
        (out,), _ = self.sd(
            prompt=self.prompt,
            image=xs,
            strength=self.strength,
            num_inference_steps=self.nsteps,
            guidance_scale=self.cfg,
            negative_prompt=self.negative_prompt,
            return_dict=False,
            output_type="pt",  # return a Tensor
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=['latents']
        )
        # Given batch size 1, the pipeline automatically removes this dimension
        if len(out.shape) == 3:
            return out.unsqueeze(0)
        return out


if __name__ == "__main__":
    parser = util.get_parser()
    parser.add_argument("-gs", "--guidance-strength", type=float, default=100,
                        help="Multiplier for the L2 guidance")
    args = parser.parse_args()
    prompt = " ".join(args.prompt) if args.prompt is not None else None
    nprompt = " ".join(args.nprompt) if args.nprompt is not None else None

    print(f"Running with strength={args.strength}, cfg={args.cfg}, prompt={prompt}, nprompt={nprompt}")

    transform = util.eval_transform(size=512)
    test_ds = NoisyDataset(root_path=args.directory, split="test", transform=transform)

    sd = diffusers.StableDiffusionImg2ImgPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    # sd.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
    sd.set_progress_bar_config(disable=True)

    model = GuidedImageToImage(sd,
                               strength=args.strength,
                               prompt=prompt,
                               negative_prompt=nprompt,
                               cfg=args.cfg,
                               nsteps=args.nsteps,
                               guidance_strength=args.guidance_strength).to(device)

    out_dir = f"out/l2guidedsd_s={args.strength}_cfg={args.cfg}_gs={args.guidance_strength}"
    util.inference_on_dataset(model, test_ds, out_dir, batch_size=1)

