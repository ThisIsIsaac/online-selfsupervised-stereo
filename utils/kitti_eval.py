from __future__ import absolute_import, division
from rich import print, pretty
pretty.install()

import cv2
import numpy as np

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    diff = np.abs(gt - pred)
    pix_test = diff > 3
    val_test = diff > gt*0.05
    outliers = np.logical_and(pix_test, val_test)
    percent_outlier = np.sum(outliers) /outliers.size

    return percent_outlier, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def evaluate(gt_depths, pred_disps):
    """Evaluates a pretrained model using a specified test set
    """

    errors = []

    if type(gt_depths) != np.ndarray:
        gt_depths = gt_depths.detach().cpu().numpy()

    if type(pred_disps) != np.ndarray:
        pred_disps = pred_disps.detach().cpu().numpy()

    if len(gt_depths.shape) == 2:
        gt_depths = gt_depths[np.newaxis, ...]

    if len(pred_disps.shape) == 2:
        pred_disps = pred_disps[np.newaxis, ...]

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[-2], gt_depth.shape[-1] #NCHW

        pred_disp = pred_disps[i]
        ratio = gt_width / pred_disp.shape[1]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height)) * ratio

        mask = gt_depth > 0
        pred_disp = pred_disp[mask]
        gt_depth = gt_depth[mask]


        percent_outlier, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_errors(gt_depth, pred_disp)

        out = {}
        out["d_all"] = percent_outlier
        out["abs_rel"] = abs_rel
        out["sq_rel"] = sq_rel
        out["rmse"] = rmse
        out["rmse_log"] = rmse_log
        out["a1"] = a1
        out["a2"] = a2
        out["a3"] = a3

        errors.append(out)

    return errors


