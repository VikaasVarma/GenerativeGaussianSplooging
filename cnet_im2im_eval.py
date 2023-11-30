"""
Evaluates ControlNet on the test set of a dataset. Used for evaluation and comparison to other models.
"""
import diffusers
import torch

from sd_l2_guidance import GuidedImageToImage
# from baseline import ImageToImageBaseline
from noisy_dataset import NoisyDataset
import util
from util import device

if __name__ == "__main__":
    parser = util.get_parser()
    parser.add_argument("-gs", "--guidance-strength", type=float, default=100,
                        help="Multiplier for the L2 guidance")
    parser.add_argument("--controlnet", type=str, required=True, help="Path to ControlNet model")
    parser.add_argument("--control-strength", type=float, default=1)
    args = parser.parse_args()
    prompt = " ".join(args.prompt) if args.prompt is not None else None
    nprompt = " ".join(args.nprompt) if args.nprompt is not None else None

    transform = util.eval_transform(size=512)
    test_ds = NoisyDataset(root_path=args.directory, split="test", transform=transform)

    # By default, Stable Diffusion uses the PNDM scheduler.
    # Here I use DDIM, and copy the following parameters from the default Stable Diffusion scheduler:
    scheduler = diffusers.DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                        clip_sample=False, set_alpha_to_one=False)

    controlnet = diffusers.ControlNetModel.from_single_file(args.controlnet, torch_dtype=torch.float16).to(util.device)
    sd = diffusers.StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.model,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(util.device)
    sd.scheduler = scheduler
    sd.set_progress_bar_config(disable=True)
    sd.enable_xformers_memory_efficient_attention()

    model = GuidedImageToImage(sd,
                               strength=args.strength,
                               prompt=prompt,
                               negative_prompt=nprompt,
                               cfg=args.cfg,
                               nsteps=args.nsteps,
                               guidance_strength=args.guidance_strength).to(device)


    def inference_function(xs):
        return model(xs,
                     control_image=2 * xs - 1,
                     controlnet_conditioning_scale=args.control_strength)


    out_dir = f"out/cnet_im2im_s={args.strength}_gs={args.guidance_strength}_cs={args.control_strength}"
    util.inference_on_dataset(inference_function, test_ds, out_dir, batch_size=1)
