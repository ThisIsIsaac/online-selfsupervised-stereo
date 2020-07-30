import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from colorspacious import cspace_converter
import numpy as np
import cv2

# input: unnormalized INT16
def clean_disp(disp):
    max_value = np.iinfo(disp.dtype).max

    disp[disp==np.NINF] = max_value
    disp[disp==np.inf] = max_value
    norm = plt.Normalize()

    return norm(disp)

# Todo: the contrast is too low. Tried multiplying by constant (before & after normalization) but doesn't change anything\
def save_colormap(x, path):
    x = clean_disp(x)
    _, ax = plt.subplots()

    # Get RGB values for colormap and convert the colormap in
    # CAM02-UCS colorspace.  lab[0, :, 0] is the lightness.
    rgb = cm.get_cmap("plasma")(x)[:, :, :3]
    lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)

    cv2.imwrite(path, lab)