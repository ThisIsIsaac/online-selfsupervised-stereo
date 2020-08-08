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



def get_metrics(disp, gt, max_disp, datatype, save_path, disp_path=None, gt_path=None, obj_map_path=None, debug=False, ABS_THRESH=3.0, REL_THRESH=0.05):
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
    # if datatype == 0 or datatype == 1:
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

    # KITTI metric
    if datatype == 2:
        if obj_map_path == None:
            raise ValueError("object map path has to be given for KITTI")


        num_errors_bg = 0.0
        num_pixels_bg = 0.0
        num_errors_fg = 0.0
        num_pixels_fg = 0.0
        num_errors_all = 0.0


        # load object map (0:background, >0:foreground)
        obj_map = (skimage.io.imread(obj_map_path).astype('int32'))
        height = disp.shape[0] # 2D: HW
        width = disp.shape[1]

        for h in range(height):
            for w in range(width):
                if bool(mask[h][w] == False):
                    continue
                d_gt = float(gt[h][w])
                d_est = float(disp[h][w])

                # source: devkit/cpp/evaluate_scene_flow.cpp
                # do this for every pixel:
                is_err = np.fabs(d_gt - d_est) > ABS_THRESH and (np.fabs(d_gt - d_est) / np.fabs(d_gt)) > REL_THRESH

                # backgrond
                if obj_map[h][w] == 0:
                    if is_err:
                        num_errors_bg+=1
                        num_errors_all+=1
                    num_pixels_bg+=1
                else:
                    if is_err:
                        num_errors_fg+=1
                        num_errors_all+=1
                    num_pixels_fg+=1

        num_pixels_all = num_pixels_bg + num_pixels_fg
        
        metrics["num_errors_bg"] = num_errors_bg
        metrics["num_pixels_bg"] = num_pixels_bg
        metrics["num_errors_fg"] = num_errors_fg
        metrics["num_pixels_fg"] = num_pixels_fg
        metrics["num_errors_all"] = num_errors_all
        metrics["num_pixels_all"] = num_pixels_all

        
        metrics["d_all"] = num_errors_all / num_pixels_all
        metrics["d_bg"] = num_errors_bg / num_pixels_bg
        metrics["d_fg"] = num_errors_fg / num_pixels_fg

    else:
        num_pixels_all = 0.0
        num_errors_all = 0.0

        # load object map (0:background, >0:foreground)
        height = disp.shape[0]  # 2D: HW
        width = disp.shape[1]

        for h in range(height):
            for w in range(width):
                if mask[h][w] == False:
                    continue
                d_gt = gt[h][w]
                d_est = disp[h][w]

                # source: devkit/cpp/evaluate_scene_flow.cpp
                # do this for every pixel:
                is_err = np.fabs(d_gt - d_est) > ABS_THRESH and np.fabs(d_gt - d_est) / np.fabs(d_gt) > REL_THRESH

                if is_err:
                    num_errors_all+=1
                num_pixels_all+=1

        metrics["d_all"] = num_errors_all / num_pixels_all

    metrics["avgerr"] = avgerr
    metrics["rms"] = rms
    
    if debug:
        # interpolated_raw_disp_path = os.path.join(save_path, "i_disp.png")
        # cv2.imwrite(interpolated_raw_disp_path, disp)

        # raw_gt_path = os.path.join(save_path, "raw_gt.png")
        # cv2.imwrite(raw_gt_path, gt)
        
        metrics_path = os.path.join(save_path, "metrics.txt")
        with open(metrics_path, "w") as file:
            for key,val in metrics.items():
                file.write(key + ": " + str(float(val)) + "\n")
    
    return metrics
