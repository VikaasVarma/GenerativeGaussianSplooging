import noisy_dataset
import torch.utils.data as dutils


class ControlNetDataset(dutils.Dataset):
    """A wrapper for NoisyDataset, providing data in the format expected by ControlNet"""

    def __init__(self, ds: noisy_dataset.NoisyDataset, prompt: str, permute: bool = True):
        self.ds = ds
        self.permute = permute
        self.prompt = prompt

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        render_im, gt_im = self.ds[item]

        gt_im = 2 * gt_im - 1
        render_im = 2 * render_im - 1

        if self.permute:
            gt_im = gt_im.permute(1, 2, 0)
            render_im = render_im.permute(1, 2, 0)

        return dict(jpg=gt_im, hint=render_im, txt=self.prompt)  # provide in expected format
