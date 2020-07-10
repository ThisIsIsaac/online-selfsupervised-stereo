import warnings
warnings.filterwarnings("ignore")
import numpy as np
from utils.readpfm import readPFM
import cv2
import skimage.io
import os


def get_metrics(disp, gt, max_disp, datatype, obj_map_path=None, ABS_THRESH=3.0, REL_THRESH=0.05):
    mask = gt != np.inf
    metrics = {}
    if disp.shape != gt.shape:
        ratio = float(gt.shape[1]) / disp.shape[1]
        disp = cv2.resize(disp, (gt.shape[1], gt.shape[0])) * ratio

    disp[disp > max_disp] = max_disp

    errmap = np.abs(gt - disp) * mask

    # average error
    avgerr = errmap[mask].mean()

    # root mean squared error
    rms = np.sqrt((errmap[mask] ** 2).mean())

    # Middlebury and ETH15 both use Middlebury evaluation metric
    if datatype == 0 or datatype == 1:
        bad5map = (errmap > 1) * mask
        bad5 = bad5map[mask].sum() / float(mask.sum()) * 100

        bad10map = (errmap > 2) * mask
        bad10 = bad10map[mask].sum() / float(mask.sum()) * 100

        bad20map = (errmap > 4) * mask
        bad20 = bad20map[mask].sum() / float(mask.sum()) * 100

        metrics["bad5"] = bad5
        metrics["bad10"] = bad10
        metrics["bad20"] = bad20

    # KITTI metric
    elif datatype == 2:
        if obj_map_path == None:
            raise ValueError("object map path has to be given for KITTI")


        num_errors_bg = 0
        num_pixels_bg = 0
        num_errors_fg = 0
        num_pixels_fg = 0
        num_errors_all = 0


        # load object map (0:background, >0:foreground)
        obj_map = (skimage.io.imread(obj_map_path).astype('int32'))
        height = disp.shape[0] # 2D: HW
        width = disp.shape[1]

        for h in range(height):
            for w in range(width):
                d_gt = gt[h][w]
                d_est = disp[h][w]

                # source: devkit/cpp/evaluate_scene_flow.cpp
                # do this for every pixel:
                is_err = np.fabs(d_gt - d_est) > ABS_THRESH and np.fabs(d_gt - d_est) / np.fabs(d_gt) > REL_THRESH

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
        metrics["d_all"] = num_errors_all / num_pixels_all
        metrics["d_bg"] = num_errors_bg / num_pixels_bg
        metrics["d_fg"] = num_errors_fg / num_pixels_fg

    metrics["avgerr"] = avgerr
    metrics["rms"] = rms
    return metrics
