"""
Training script adapted from openai/guided-diffusion to condition on images
"""

import argparse
import torch.utils.data as dutils
import noisy_dataset
import torch

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


class OpenAIDataset(dutils.Dataset):
    """A wrapper for NoisyDataset which uses """

    def __init__(self, path: str, split: str, size: int = 256):
        self.ds = noisy_dataset.NoisyDataset(
            root_path=path,
            split=split,
            transform=noisy_dataset.get_transform(size=size, crop_size=0.6)
        )
        self.size = size

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        render_im, gt_im = self.ds[item]
        # Normalise both to [-1, 1] as expected by OpenAI's codebase
        gt_im = 2 * gt_im - 1
        render_im = 2 * render_im - 1

        # Return in required format
        return gt_im, {
            "concat_cond": render_im
        }


def get_dataloader(args):
    ds = OpenAIDataset(
        path=args.data_dir,
        split="train",
        size=args.image_size
    )
    return dutils.DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        in_channels=6
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = get_dataloader(args)
    # data = load_data(
    #     data_dir=args.data_dir,
    #     batch_size=args.batch_size,
    #     image_size=args.image_size,
    #     class_cond=args.class_cond,
    # )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
