# from __future__ import print_function
from rich import print
from rich import pretty
pretty.install()
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
from trainer import _init_fn

def main():
    parser = argparse.ArgumentParser(description='HSM-Net')
    parser.add_argument('--maxdisp', type=int, default=384,
                        help='maxium disparity')
    parser.add_argument('--name', default='name')
    parser.add_argument('--database', default='/data/private',
                        help='data path')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=18,
                        # when maxdisp is 768, 18 is the most you can fit in 2 V100s (with syncBN on)
                        help='samples per batch')
    parser.add_argument('--loadmodel', default=None,
                        help='weights path')
    parser.add_argument('--log_dir', default="/data/private/logs/high-res-stereo")
    # parser.add_argument('--savemodel', default=os.path.join(os.getcwd(),'/trained_model'),
    #                     help='save path')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--val_epoch', type=int, default=2)
    parser.add_argument('--save_epoch', type=int, default=1)
    parser.add_argument("--val", action="store_true", default=False)
    parser.add_argument("--save_numpy", action="store_true", default=False)
    parser.add_argument("--testres", type=float, default=1.8)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--use_pseudoGT", default=False, action="store_true")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--lr_decay", default=2, type=int)
    parser.add_argument("--gpu", default=[0], nargs="+")

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.manual_seed(args.seed)  # set again
    torch.cuda.manual_seed(args.seed)
    batch_size = args.batch_size
    scale_factor = args.maxdisp / 384.  # controls training resolution
    args.name = args.name + "_" + time.strftime('%l:%M%p_%Y%b%d').strip(" ")
    gpu = []
    for i in args.gpu:
        gpu.append(int(i))
    args.gpu=gpu

    # all_left_img, all_right_img, all_left_disp, all_right_disp, left_val, right_val, disp_val_L, disp_val_R = ls.hr_dataloader(
    #     '%s/HR-VS/trainingF' % args.database, val=args.val)
    # loader_carla = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, right_disparity=all_right_disp,
    #                                 rand_scale=[0.225, 0.6 * scale_factor], rand_bright=[0.8, 1.2], order=2)
    #
    # val_loader_carla = DA.myImageFloder(left_val, right_val, disp_val_L, right_disparity=disp_val_R,
    #                                     rand_scale=[0.225, 0.6 * scale_factor], rand_bright=[0.8, 1.2], order=2)
    #
    # all_left_img, all_right_img, all_left_disp, all_right_disp, left_val, right_val, disp_val_L, disp_val_R = ls.mb_dataloader(
    #     '%s/Middlebury/mb-ex-training/trainingF' % args.database, val=args.val)  # mb-ex
    # loader_mb = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, right_disparity=all_right_disp,
    #                              rand_scale=[0.225, 0.6 * scale_factor], rand_bright=[0.8, 1.2], order=0)
    # val_loader_mb = DA.myImageFloder(left_val, right_val, disp_val_L, right_disparity=disp_val_R,
    #                                  rand_scale=[0.225, 0.6 * scale_factor], rand_bright=[0.8, 1.2], order=0)
    #
    # all_left_img, all_right_img, all_left_disp, all_right_disp, val_left_img, val_right_img, val_left_disp, val_right_disp = lt.scene_dataloader(
    #     '%s/SceneFlow/' % args.database, val=args.val)
    # loader_scene = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, right_disparity=all_right_disp,
    #                                 rand_scale=[0.9, 2.4 * scale_factor], order=2)
    # val_loader_scene = DA.myImageFloder(val_left_img, val_right_img, val_left_disp, right_disparity=val_right_disp,
    #                                     rand_scale=[0.9, 2.4 * scale_factor], order=2)

    all_left_img, all_right_img, all_left_disp, left_val, right_val, disp_val_L = lk15.dataloader(
        '%s/KITTI2015/data_scene_flow/training/' % args.database,
        val=args.val)




    loader_kitti15 = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, rand_scale=[0.9, 2.4 * scale_factor],
                                      order=0, use_pseudoGT=args.use_pseudoGT, entropy_threshold=args.threshold)
    val_loader_kitti15 = DA.myImageFloder(left_val, right_val, disp_val_L, is_validation=True, testres=3.3)

    #
    # all_left_img, all_right_img, all_left_disp, left_val, right_val, disp_val_L = lk12.dataloader(
    #     '%s/KITTI2012/data_stereo_flow/training/' % args.database, val=args.val)
    # loader_kitti12 = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, rand_scale=kitti_scale_range,
    #                                   order=0)
    # val_loader_kitti12 = DA.myImageFloder(left_val, right_val, disp_val_L, rand_scale=kitti_scale_range,
    #                                       order=0)
    #
    # # ! Gengshan didn't tell me to change ETH explicitly, just change kitti. Although he didn't mention explicitly, it may be better to chagne ETH as well, so will go a head
    # all_left_img, all_right_img, all_left_disp, left_val, right_val, disp_val_L = ls.eth_dataloader(
    #     '%s/ETH3D/low-res-stereo/train/' % args.database, val=args.val)
    # loader_eth3d = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, rand_scale=kitti_scale_range, order=0)
    # val_loader_eth3d = DA.myImageFloder(left_val, right_val, disp_val_L, rand_scale=kitti_scale_range, order=0)

    # * Gengshan told me to double the loader_kitti15's proportion by multiplying 2
    # data_inuse = torch.utils.data.ConcatDataset(
    #     [loader_carla] * 40 + [loader_mb] * 500 + [loader_scene] + [loader_kitti15] * 2 + [loader_kitti12] * 80 + [
    #         loader_eth3d] * 1000)
    train_data_inuse = loader_kitti15
    val_data_inuse = val_loader_kitti15

    #! For internal bug in Pytorch, if you are going to set num_workers >0 in one dataloader, it must also be set to
    #! n >0 for the other data loader as well (ex. 1 for valLoader and 10 for trainLoader)
    ValImgLoader = torch.utils.data.DataLoader(val_data_inuse, drop_last=False, batch_size=4,
                                               shuffle=False, worker_init_fn=_init_fn, num_workers=4)

    TrainImgLoader = torch.utils.data.DataLoader(
        train_data_inuse,
        batch_size=batch_size, shuffle=True,  drop_last=True, worker_init_fn=_init_fn, num_workers=args.batch_size) #, , worker_init_fn=_init_fn

    print('%d batches per epoch' % (len(train_data_inuse) // batch_size))

    model = hsm(args.maxdisp, clean=False, level=1)

    gpus = [0, 1, 2, 3]
    if len(gpus) > 1:
        from sync_batchnorm.sync_batchnorm import convert_model
        model = nn.DataParallel(model, device_ids=gpus)
        model = convert_model(model)
    else:
        model = nn.DataParallel(model, device_ids=gpus)

    model.cuda()

    # load model
    if args.loadmodel is not None:
        print("loading pretrained model: " + str(args.loadmodel))
        pretrained_dict = torch.load(args.loadmodel)
        pretrained_dict['state_dict'] = {k: v for k, v in pretrained_dict['state_dict'].items() if ('disp' not in k)}
        model.load_state_dict(pretrained_dict['state_dict'], strict=False)

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    log = logger.Logger(args.log_dir, args.name, save_numpy=args.save_numpy)
    # log.watch(model)
    total_iters = 0
    val_iters = 0

    save_path = os.path.join(args.log_dir, os.path.join(args.name, "saved_model"))
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        total_train_loss = 0
        train_score_accum_dict = {} # accumulates scores throughout a batch to get average score
        train_score_accum_dict["num_scored"] = 0
        adjust_learning_rate(optimizer, args.lr, args.lr_decay, epoch, args.epochs, decay_rate=0.1)

        print('Epoch %d / %d' % (epoch, args.epochs))

        # SAVE
        if epoch % args.save_epoch == 0:
            print("saving weights at epoch: " + str(epoch))
            savefilename = os.path.join(save_path, 'ckpt_' + str(total_iters) + '.tar')

            torch.save({
                'iters': total_iters,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss / len(TrainImgLoader),
                "optimizer": optimizer.state_dict()
            }, savefilename)

        ## val ##
        if epoch == 1 or epoch % args.val_epoch == 0:
            print("validating at epoch: " + str(epoch))
            val_score_accum_dict = {} # accumulates scores throughout a batch to get average score
            for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(ValImgLoader):


                vis, scores_list, maps = val_step(model, imgL_crop, imgR_crop, disp_crop_L, args.maxdisp, args.testres)

                for score in scores_list:
                    for tag, val in score.items():
                        log.scalar_summary("val/" + tag + "_batch", val, val_iters)

                        if tag not in val_score_accum_dict.keys():
                            val_score_accum_dict[tag] = 0
                        val_score_accum_dict[tag] += val

                for tag, err_map in maps[0].items():
                    log.image_summary("val/"+tag, err_map, val_iters)

                log.image_summary('val/left', imgL_crop[0:1], val_iters)
                log.image_summary('val/right', imgR_crop[0:1], val_iters)
                log.disp_summary('val/gt0', disp_crop_L[0:1], val_iters)  # <-- GT disp
                log.entp_summary('val/entropy', vis['entropy'], val_iters)
                log.disp_summary('val/output3', vis['output3'][0], val_iters)

                val_iters += 1

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):

            is_scoring = total_iters%10 == 0

            loss, vis, scores_list, maps = train_step(model, optimizer, imgL_crop, imgR_crop, disp_crop_L, args.maxdisp, is_scoring=is_scoring)

            total_train_loss += loss

            if is_scoring:
                log.scalar_summary('train/loss_batch', loss, total_iters)
                for score in scores_list:
                    for tag, val  in score.items():
                        log.scalar_summary("train/" + tag + "_batch", val, total_iters)

                        if tag not in train_score_accum_dict.keys():
                            train_score_accum_dict[tag] = 0
                        train_score_accum_dict[tag] += val
                        train_score_accum_dict["num_scored"] += imgL_crop.shape[0]

                for tag, err_map in maps[0].items():
                    log.image_summary("train/"+tag, err_map, total_iters)

            if total_iters % 10 == 0:
                log.image_summary('train/left', imgL_crop[0:1], total_iters)
                log.image_summary('train/right', imgR_crop[0:1], total_iters)
                log.disp_summary('train/gt0', disp_crop_L[0:1], total_iters) # <-- GT disp
                log.entp_summary('train/entropy', vis['entropy'][0:1], total_iters)
                log.disp_summary('train/output3', vis['output3'][0:1], total_iters)

            total_iters += 1



        log.scalar_summary('train/loss', total_train_loss / len(TrainImgLoader), epoch)
        for tag, val in train_score_accum_dict.items():
            log.scalar_summary("train/" + tag + "_avg", val / train_score_accum_dict["num_scored"], epoch)

        torch.cuda.empty_cache()
    # Save final checkpoint
    print("Finished training!\n Saving the last checkpoint...")
    savefilename = os.path.join(save_path, 'final' + '.tar')

    torch.save({
        'iters': total_iters,
        'state_dict': model.state_dict(),
        'train_loss': total_train_loss / len(TrainImgLoader),
        "optimizer": optimizer.state_dict()
    }, savefilename)

if __name__ == '__main__':
    main()
