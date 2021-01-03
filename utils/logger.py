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
import matplotlib
import matplotlib.pyplot as plt
import io
from PIL import Image
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)



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

        if images.dtype == bool:
            images = images.astype(float)
        images = (images - images.min()) / (images.max() - images.min())

        if len(images.shape) == 4:
            self.writer.add_images(tag, images, step)

        if len(images.shape) == 3:
            self.writer.add_image(tag, images, step)

    def heatmap_summary(self, tag, values, step):
        light_jet = cmap_map(lambda x: x / 2 + 0.5, matplotlib.cm.jet)
         # = plt.figure()
        fig=plt.imshow(values, cmap=light_jet)
        plt.colorbar(fig)
        plt.title(tag)
        # img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        # buf = io.BytesIO()
        # plt.savefig(buf, format='png')
        # buf.seek(0)

        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure

        fig = Figure()
        canvas = FigureCanvas(fig)
        canvas.draw()  # draw the canvas, cache the renderer

        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
        self.writer.add_image(tag, image, step)



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
        if type(values) == list:
            values = np.array(values)
        self.writer.add_histogram(tag, values, global_step=step, bins=bins)
        self.writer.flush()


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

