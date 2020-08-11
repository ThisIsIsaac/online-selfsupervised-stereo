"""
File: logger.py
Modified by: Senthil Purushwalkam
Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
Email: spurushw<at>andrew<dot>cmu<dot>edu
Github: https://github.com/senthilps8
Description:
"""
import torch
import wandb
from torch.autograd import Variable
import numpy as np
import scipy.misc
from utils import disp_converter
import os
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


class WandbLogger(object):
    def __init__(self, args):
        """Create a summary writer logging to log_dir."""
        if args.logname == None:
            args.logname = 'temp'
        self.name = args.logname
        wandb.init(name=self.name, save_code=True, magic=True, config=args)
        # if name is not None:
        #     try:
        #         os.makedirs(os.path.join(log_dir, name))
        #     except:
        #         pass
        #     self.writer = tf.summary.FileWriter(os.path.join(log_dir, name),
        #                                         filename_suffix=name)
        # else:
        #     self.writer = tf.summary.FileWriter(log_dir, filename_suffix=name)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        wandb.log({tag:value}, step=step)
        # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        # self.writer.add_summary(summary, step)

    def image_summary(self, tag, img, step, caption=None):
        """Log an image."""
        img = self.to_jpg(img)

        color = disp_converter.convert_to_colormap(img)

        data = {tag: wandb.Image(img, caption=caption), tag+"_color": wandb.Image(color, caption=caption)}

        wandb.log(data,  step=step)

    def diff_summary(self, tag, gt, pred, step, caption=None):
        gt = self.to_jpg(gt)
        pred = self.to_jpg(self)

        diff, false_negative_map, false_positive_map = disp_converter.get_diffs(gt, pred)

        self.image_summary(tag, diff, step, caption=caption)
        self.image_summary(tag + "_false_negative", false_negative_map, step, caption=caption)
        self.image_summary(tag + "_false_positive", false_positive_map, step, caption=caption)

    def to_jpg(self, x):
        x = self.to_np(x)
        return (x*256).astype("uint16")

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        wandb.log({tag:wandb.Histogram(np_histogram=np.histogram(values, bins=bins))})

    def to_np(self, x):
        return x.data.cpu().numpy()

    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def model_param_histo_summary(self, model, step):
        """log histogram summary of model's parameters
        and parameter gradients
        """
        for tag, value in model.named_parameters():
            if value.grad is None:
                continue
            tag = tag.replace('.', '/')
            tag = self.name+'/'+tag
            self.histo_summary(tag, self.to_np(value), step)
            self.histo_summary(tag+'/grad', self.to_np(value.grad), step)

    def watch(self, model):
        wandb.watch(model)
