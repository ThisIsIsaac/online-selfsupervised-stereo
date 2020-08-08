import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from colorspacious import cspace_converter
import numpy as np
import cv2

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

# def interpolate(colormap, x):
def interpolate(x):
    a = (x*255.0).astype(np.int16)
    f = x*255.0 - a
    return f

def clean_disp(disp):
    disp[disp==np.NINF] = 0
    disp[disp==np.inf] = 0
    norm = plt.Normalize()

    disp = interpolate(disp)
    disp = norm(disp)

    return disp

# Todo: the contrast is too low. Tried multiplying by constant (before & after normalization) but doesn't change anything\
def convert_to_colormap(x):
    x = clean_disp(x)
    rgb = cm.get_cmap("plasma")(x)[:, :, :3]
    lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)

    return lab

def get_diffs(gt, pred):
    gt_invalid = np.logical_or(gt == np.inf, gt == np.NINF)

    diff = gt - pred

    # when pred has lower depth than GT
    false_negative_map = np.clip(diff, 0, None)

    # when pred has higher depth than GT
    false_positive_map = np.abs(np.clip(diff, None, 0))

    false_negative_map[gt_invalid] = 0
    false_positive_map[gt_invalid] = 0
    diff[gt_invalid] = 0

    return diff, false_negative_map, false_positive_map

