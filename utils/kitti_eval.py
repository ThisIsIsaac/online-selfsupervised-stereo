from __future__ import absolute_import, division
from rich import print, pretty
pretty.install()

import cv2
import numpy as np

def compute_errors(gt, pred):
    NA_mask = gt <= 0
    thresh_map = np.maximum((gt / pred), (pred / gt))
    diff_map = (gt - pred)
    pix_test = diff_map > 3
    val_test = diff_map > gt * 0.05
    outliers_map = np.logical_and(pix_test, val_test)

    abs_rel_map = np.abs(gt - pred) / gt

    diff_map[NA_mask] = 0
    abs_rel_map[NA_mask] = 0
    outliers_map[NA_mask] = 0



    mask = gt > 0
    pred = pred[mask]
    gt = gt[mask]
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1_map = thresh < 1.25
    a1 = a1_map.mean()

    a2_map = thresh < 1.25**2
    a2 = a2_map.mean()

    a3_map = thresh < 1.25**3
    a3 = a3_map.mean()

    diff = (gt - pred)
    rmse = np.sqrt((diff ** 2).mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.abs(gt - pred) / gt
    abs_rel = np.mean(abs_rel)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    diff = np.abs(gt - pred)
    pix_test = diff > 3
    val_test = diff > gt*0.05
    outliers = np.logical_and(pix_test, val_test)
    percent_outlier = np.sum(outliers) /outliers.size

    return ((percent_outlier, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3),
            (outliers_map, abs_rel_map, diff_map, thresh_map))


def evaluate(gt_depths, pred_disps):
    """Evaluates a pretrained model using a specified test set
    """

    errors = []
    error_maps = []

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




        scores, maps = compute_errors(gt_depth, pred_disp)


        percent_outlier, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = scores

        # resized_maps = []
        # for m in maps:
        #     resized_maps.append(m.reshape(gt_height, gt_width))

        outliers_map, abs_rel_map, diff_map, thresh_map = maps


        out = {}
        out["d_all"] = percent_outlier
        # out["abs_rel"] = abs_rel
        # out["sq_rel"] = sq_rel
        out["rmse"] = rmse
        # out["rmse_log"] = rmse_log
        # out["a1"] = a1
        # out["a2"] = a2
        # out["a3"] = a3
        errors.append(out)

        out_maps = {}
        out_maps["outliers_map"] = outliers_map
        out_maps["abs_rel_map"] = abs_rel_map
        out_maps["diff_map"] = diff_map
        # out_maps["thresh_map"] = thresh_map

        error_maps.append(out_maps)

    return errors, error_maps


