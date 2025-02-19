import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from colorspacious import cspace_converter
import numpy as np
import cv2
import os
import glob

def save_disp_as_colormap(disp, path):
    if len(disp.shape) == 3:
        disp = disp[0]
    values = (disp * (256)).astype("uint16")
    values = convert_to_colormap(values)
    # values = np.transpose(values, axes=[2, 0, 1])

    # values = (values - values.min()) / (values.max() - values.min())
    values = values / values.max()
    values = (values *512)
    print(cv2.imwrite(path, values))
    cv2.imwrite(path, values)

# source: https://github.com/mrharicot/monodepth/issues/118#issuecomment-369582311
def disp_to_depth(pred_disp, img_width, dataset="kitti"):
    if dataset == "kitti":
        return 0.54 * 721 / (img_width * pred_disp)

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
    # lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)
    # lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return rgb

def get_diffs(gt, pred):
    gt_invalid = np.logical_or(gt == np.inf, gt == 0)

    diff = gt - pred

    # when pred has lower depth than GT
    false_negative_map = np.clip(diff, 0, None)

    # when pred has higher depth than GT
    false_positive_map = np.abs(np.clip(diff, None, 0))

    false_negative_map[gt_invalid] = 0
    false_positive_map[gt_invalid] = 0
    diff[gt_invalid] = 0

    return diff, false_negative_map, false_positive_map

