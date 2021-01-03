# from __future__ import print_function
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from rich import print
from rich import pretty
pretty.install()
from PIL import Image
from rich import traceback
traceback.install()
import pdb
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from models import hsm
from utils import logger
import plotly.express as px
from utils import get_metrics
from collections import Counter
from utils import kitti_eval
import cv2
from dataloader import listfiles as ls
from dataloader import listsceneflow as lt
from dataloader import KITTIloader2015 as lk15
from dataloader import KITTIloader2012 as lk12
from dataloader import MiddleburyLoader as DA
import math
torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
from dataloader.unlabeled_loader import kitti_raw_loader
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from trainer import *
from datetime import datetime
import torchvision


def _init_fn(worker_id):
    np.random.seed()
    random.seed()

def main():
    parser = argparse.ArgumentParser(description='HSM-Net')
    parser.add_argument('--maxdisp', type=int, default=384,
                        help='maxium disparity')
    parser.add_argument('--name', default='name')
    parser.add_argument('--database', default='/data/private',
                        help='data path')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--val_batch_size', type=int, default=1,
                        help='samples per batch')
    parser.add_argument('--loadmodel', default=None,
                        help='weights path')
    parser.add_argument('--log_dir', default="/data/private/logs/high-res-stereo")
    parser.add_argument("--testres", default=[0], nargs="+")
    parser.add_argument("--no_aug",default=False, action="store_true")

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.manual_seed(args.seed)  # set again
    torch.cuda.manual_seed(args.seed)
    args.name = args.name + "_" + time.strftime('%l:%M%p_%Y%b%d').strip(" ")
    testres = []
    for i in args.testres:
        testres.append(float(i))
    args.testres=testres

    all_left_img, all_right_img, all_left_disp, left_val, right_val, disp_val_L = lk15.dataloader(
        '%s/KITTI2015/data_scene_flow/training/' % args.database, val=True)

    left_val = [left_val[3]]
    right_val = [right_val[3]]
    disp_val_L = [disp_val_L[3]]

    # all_l = all_left_disp + left_val
    # all_r = all_right_img + right_val
    # all_d = all_left_disp + disp_val_L

    # correct_shape = (1242, 375)
    # for i in range(len(all_l)):
    #     l = np.array(Image.open(all_l[i]).convert("RGB"))
    #     r = np.array(Image.open(all_r[i]).convert("RGB"))
    #     d = Image.open(all_d[i])
    #     if l.shape != (375, 1242, 3):
    #
    #         l2 = cv2.resize(l, correct_shape, interpolation=cv2.INTER_CUBIC)
    #         r2 = cv2.resize(r, correct_shape, interpolation=cv2.INTER_CUBIC)
    #         d2 = np.array(torchvision.transforms.functional.resize(d, [375, 1242]))
    #         # d = np.stack([d, d, d], axis=-1)
    #         # d2 = cv2.resize(d.astype("uint16"), correct_shape)
    #
    #         cv2.imwrite(all_l[i], cv2.cvtColor(l2, cv2.COLOR_RGB2BGR))
    #         cv2.imwrite(all_r[i], cv2.cvtColor(r2, cv2.COLOR_RGB2BGR))
    #         cv2.imwrite(all_d[i], d2)


        # cv2.resize(l,())
    model = hsm(args.maxdisp, clean=False, level=1)
    model.cuda()

    # load model
    print("loading pretrained model: " + str(args.loadmodel))
    pretrained_dict = torch.load(args.loadmodel)
    pretrained_dict['state_dict'] = {k: v for k, v in pretrained_dict['state_dict'].items() if ('disp' not in k)}
    model = nn.DataParallel(model, device_ids=[0])
    model.load_state_dict(pretrained_dict['state_dict'], strict=False)

    name = "val_at_many_res" + "_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log = logger.Logger(args.log_dir, name)
    val_sample_count = 0
    for res in args.testres:

        val_loader_kitti15 = DA.myImageFloder(left_val, right_val, disp_val_L, is_validation=True, testres=res)
        ValImgLoader = torch.utils.data.DataLoader(val_loader_kitti15, drop_last=False, batch_size=args.val_batch_size,
                                                   shuffle=False, worker_init_fn=_init_fn,
                                                   num_workers=0)
        print("================ res: " + str(res) + " ============================")
        ## val ##
        val_score_accum_dict = {}
        val_img_idx = 0
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(ValImgLoader):
            vis, scores_list, err_map_list = val_step(model, imgL_crop, imgR_crop, disp_crop_L, args.maxdisp, res)

            for score, err_map in zip(scores_list, err_map_list):
                for (score_tag, score_val), (map_tag, map_val) in zip(score.items(), err_map.items()):
                    log.scalar_summary("val/im_" + str(val_img_idx) + "/" + str(res) + "/"+ score_tag, score_val, val_sample_count)
                    log.image_summary("val/" + str(res) + "/"+ map_tag, map_val, val_sample_count)

                    if score_tag not in val_score_accum_dict.keys():
                        val_score_accum_dict[score_tag] = 0
                    val_score_accum_dict[score_tag]+=score_val
                    print("res: " + str(res) + " " + score_tag + ": " + str(score_val))

                val_img_idx+=1
                val_sample_count += 1

                log.image_summary('val/left', imgL_crop[0:1], val_sample_count)
                # log.image_summary('val/right', imgR_crop[0:1], val_sample_count)
                log.disp_summary('val/gt0', disp_crop_L[0:1], val_sample_count)  # <-- GT disp
                log.entp_summary('val/entropy', vis['entropy'], val_sample_count)
                log.disp_summary('val/output3', vis['output3'][0], val_sample_count)






if __name__ == '__main__':
    main()
