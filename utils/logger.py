"""
File: logger.py
Modified by: Senthil Purushwalkam
Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
Email: spurushw<at>andrew<dot>cmu<dot>edu
Github: https://github.com/senthilps8
Description: 
"""
from utils.disp_converter import convert_to_colormap
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import numpy as np
import os
import torch
import cv2

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


class Logger(object):

    def __init__(self, log_dir, name=None, save_numpy=False):
        """Create a summary writer logging to log_dir."""
        if name is None:
            name = 'temp'
        self.name = name
        self.log_dir = os.path.join(log_dir, name)
        self.save_numpy = save_numpy
        if name is not None:
            try:
                os.makedirs(self.log_dir)
            except:
                pass
            self.writer = SummaryWriter(log_dir=self.log_dir,
                                        filename_suffix=name)
        else:
            self.writer = SummaryWriter(log_dir=self.log_dir, filename_suffix=name)

        print("Logging files are saved in: " + self.log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        if type(images) != np.ndarray:
            images = images.detach().cpu().numpy()

        # save_as_png=True #! <--- for debugging. Remove later
        # if save_as_png:
        #     name = tag.replace("/", "_")
        #     images_png = images
        #     # images_png = (images * 256).astype("uint16")
        #     if len(images.shape) == 4:
        #         images_png = images_png[0]
        #         images_png = (images_png * 256).astype("uint16")
        #     if len(images.shape) ==3:
        #         images_png = np.transpose(images_png, axes=[1, 2, 0])
        #     cv2.imwrite(os.path.join(self.log_dir, name + "_" + str(step) + ".png"), images_png)
        # images = images.astype(np.float) / 255
        images = (images - images.min()) / (images.max() - images.min())

        if len(images.shape) == 4:
            self.writer.add_images(tag, images, step)

        if len(images.shape) == 3:
            self.writer.add_image(tag, images, step)


    def disp_summary(self, tag, values, step):

        if type(values) != np.ndarray:
            values = values.detach().cpu().numpy()

        if len(values.shape) ==3:
            values = values[0]

        if self.save_numpy:
            name = tag.replace("/", "_")
            np.save(os.path.join(self.log_dir, name + "_" + str(step)), values)

        values = (values * 256).astype("uint16")
        values = convert_to_colormap(values)
        values = np.transpose(values, axes=[2, 0, 1])

        images = (values - values.min()) / (values.max() - values.min())

        if len(images.shape) == 4:
            self.writer.add_images(tag, images, step)

        if len(images.shape) == 3:
            self.writer.add_image(tag, images, step)


    def entp_summary(self, tag, values, step):
        if type(values) != np.ndarray:
            values = values.detach().cpu().numpy()
        if len(values.shape) ==3:
            values = values[0]

        values = (values/values.min() * 256).astype("uint16")
        values = convert_to_colormap(values)
        values = np.transpose(values, axes=[2, 0, 1])
        images = (values - values.min()) / (values.max() - values.min())

        if self.save_numpy:
            name = tag.replace("/", "_")
            np.save(os.path.join(self.log_dir, name + "_" + str(step)), values)

        if len(images.shape) == 2:
            images = images[np.newaxis, ...]

        self.writer.add_image(tag, images, step)


    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        self.writer.add_histogram(tag, values, step, bins=bins)


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

