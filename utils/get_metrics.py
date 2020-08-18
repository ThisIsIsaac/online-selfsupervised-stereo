import warnings
warnings.filterwarnings("ignore")
import numpy as np
from utils.readpfm import readPFM
import cv2
import skimage.io
import os
import matplotlib.pyplot as plt
from subprocess import call
import subprocess
import pickle



def get_metrics(gt, disp, max_disp):
    mask = np.logical_and(gt != np.inf, gt > 0)
    metrics = {}
    if disp.shape != gt.shape:
        raise ValueError("pred and GT must be of the same dimensions")

    disp[disp > max_disp] = max_disp

    errmap = np.abs(gt - disp) * mask

    # average error
    avgerr = errmap[mask].mean()

    # root mean squared error
    rms = np.sqrt((errmap[mask] ** 2).mean())

    # Middlebury and ETH15 both use Middlebury evaluation metric
    bad5map = (errmap > 0.5) * mask
    bad5 = bad5map[mask].sum() / float(mask.sum()) * 100

    bad10map = (errmap > 1) * mask
    bad10 = bad10map[mask].sum() / float(mask.sum()) * 100

    bad20map = (errmap > 2) * mask
    bad20 = bad20map[mask].sum() / float(mask.sum()) * 100

    bad40map = (errmap > 4) * mask
    bad40 = bad40map[mask].sum() / float(mask.sum()) * 100

    metrics["bad5"] = bad5
    metrics["bad10"] = bad10
    metrics["bad20"] = bad20
    metrics["bad40"] = bad40
    metrics["avgerr"] = avgerr
    metrics["rms"] = rms

    return metrics
