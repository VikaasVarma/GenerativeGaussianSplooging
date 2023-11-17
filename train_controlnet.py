"""
Code for training ControlNet for the de-splatting task.
Adapted from the official guide: https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md
"""
import sys
sys.path.append("ControlNet")  # workaround...
import noisy_dataset
import lightning as pl
import torch.utils.data as dutils
from torch.utils.data import DataLoader
import torch
from ControlNet.cldm.logger import ImageLogger
from ControlNet.cldm.model import create_model, load_state_dict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True, help="Root path for noisy dataset")
# '../models/control-sd-v1-5.ckpt'
parser.add_argument("-m", "--model-path", type=str, required=True, help="ControlNet model path")
parser.add_argument("-b", "--batch-size", type=int, default=4, help="Batch size")
# './ControlNet/models/cldm_v15.yaml' or without ControlNet
parser.add_argument("-c", "--config", type=str, default="models/cldm_v15.yaml", help="Path to ControlNet config")
parser.add_argument("-g", "--accum-grad", type=int, default=1, help="Gradient accumulation in batches")
parser.add_argument("-a", "--accelerator", type=str, default="gpu", help="Pytorch Lightning accelerator")
args = parser.parse_args()


class ControlNetDataset(dutils.Dataset):
    """A wrapper for NoisyDataset which uses """
    def __init__(self, ds: noisy_dataset.NoisyDataset):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        render_im, gt_im = self.ds[item]
        # Normalise target to [-1, 1] as expected by ControlNet codebase
        gt_im = 2 * gt_im - 1

        gt_im = gt_im.permute(1, 2, 0)
        render_im = render_im.permute(1, 2, 0)
        
        return dict(jpg=gt_im, hint=render_im, txt="")  # provide in expected format


# Configs
resume_path = args.model_path
batch_size = args.batch_size
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(args.config).cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
transform = noisy_dataset.get_transform(size=512)
noisy_ds = noisy_dataset.NoisyDataset(root_path=args.dataset, split="train", transform=transform)
dataset = ControlNetDataset(noisy_ds)
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(accelerator=args.accelerator, devices=1, precision='bf16', accumulate_grad_batches=args.accum_grad, callbacks=[logger])


# Train!
trainer.fit(model, dataloader)
