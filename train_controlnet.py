"""
Code for training ControlNet for the de-splatting task.
Adapted from the official guide: https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md
"""
import torch.utils.data as dutils
import noisy_dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from ControlNet.cldm.logger import ImageLogger
from ControlNet.cldm.model import create_model, load_state_dict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Root path for noisy dataset")
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
        return dict(jpg=gt_im, hint=render_im, txt="")  # provide in expected format


# Configs
resume_path = './models/control-sd-v1-5.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./ControlNet/models/cldm_v15.yaml').cpu()
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
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])


# Train!
trainer.fit(model, dataloader)


