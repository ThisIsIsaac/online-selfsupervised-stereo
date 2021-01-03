# from __future__ import print_function
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
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
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16,
                        # when maxdisp is 768, 18 is the most you can fit in 2 V100s (with syncBN on)
                        help='samples per batch')
    parser.add_argument('--val_batch_size', type=int, default=4,
                        # when maxdisp is 768, 18 is the most you can fit in 2 V100s (with syncBN on)
                        help='samples per batch')
    parser.add_argument('--loadmodel', default=None,
                        help='weights path')
    parser.add_argument('--log_dir', default="/data/private/logs/high-res-stereo")
    # parser.add_argument('--savemodel', default=os.path.join(os.getcwd(),'/trained_model'),
    #                     help='save path')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--val_epoch', type=int, default=4)
    parser.add_argument('--save_epoch', type=int, default=10)
    parser.add_argument("--val", action="store_true", default=False)
    parser.add_argument("--save_numpy", action="store_true", default=False)
    parser.add_argument("--testres", type=float, default=1.8)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--use_pseudoGT", default=False, action="store_true")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--lr_decay", default=2, type=int)
    parser.add_argument("--gpu", default=[0], nargs="+")
    parser.add_argument("--no_aug",default=False, action="store_true")

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

    root_dir = "/data/private/KITTI_raw/2011_09_26/2011_09_26_drive_0013_sync"
    disp_dir = "final-768px_testres-3.3/disp"
    entp_dir = "final-768px_testres-3.3/entropy"
    mode = "image"
    image_name = "0000000040.npy" #* this is the 4th image in the validation set
    train_left, train_right, train_disp, train_entp = kitti_raw_loader(root_dir, disp_dir, entp_dir,
                                                                       mode=mode, image_name=image_name)
    train_left = train_left * args.batch_size * 16
    train_right = train_right * args.batch_size * 16
    train_disp = train_disp * args.batch_size * 16
    train_entp = train_entp * args.batch_size * 16

    all_left_img, all_right_img, all_left_disp, left_val, right_val, disp_val_L = lk15.dataloader(
        '%s/KITTI2015/data_scene_flow/training/' % args.database,
        val=args.val)

    left_val = [left_val[3]]
    right_val = [right_val[3]]
    disp_val_L = [disp_val_L[3]]

    loader_kitti15 = DA.myImageFloder(train_left, train_right, train_disp, rand_scale=[0.9, 2.4 * scale_factor],
                                      order=0, use_pseudoGT=args.use_pseudoGT, entropy_threshold=args.threshold,
                                      left_entropy=train_entp, no_aug=args.no_aug)
    val_loader_kitti15 = DA.myImageFloder(left_val, right_val, disp_val_L, is_validation=True, testres=args.testres)

    train_data_inuse = loader_kitti15
    val_data_inuse = val_loader_kitti15

    # ! For internal bug in Pytorch, if you are going to set num_workers >0 in one dataloader, it must also be set to
    # ! n >0 for the other data loader as well (ex. 1 for valLoader and 10 for trainLoader)
    ValImgLoader = torch.utils.data.DataLoader(val_data_inuse, drop_last=False, batch_size=args.val_batch_size,
                                               shuffle=False, worker_init_fn=_init_fn, num_workers=args.val_batch_size)  #

    TrainImgLoader = torch.utils.data.DataLoader(
        train_data_inuse,
        batch_size=batch_size, shuffle=True, drop_last=True, worker_init_fn=_init_fn,
        num_workers=args.batch_size)  # , , worker_init_fn=_init_fn
    print('%d batches per epoch' % (len(train_data_inuse) // batch_size))

    model = hsm(args.maxdisp, clean=False, level=1)

    if len(args.gpu) > 1:
        from sync_batchnorm.sync_batchnorm import convert_model
        model = nn.DataParallel(model, device_ids=args.gpu)
        model = convert_model(model)
    else:
        model = nn.DataParallel(model, device_ids=args.gpu)

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
    total_iters = 0
    val_sample_count = 0
    val_batch_count = 0
    save_path = os.path.join(args.log_dir, os.path.join(args.name, "saved_model"))
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        total_train_loss = 0
        train_score_accum_dict = {}  # accumulates scores throughout a batch to get average score
        train_score_accum_dict["num_scored"] = 0
        adjust_learning_rate(optimizer, args.lr, args.lr_decay, epoch, args.epochs, decay_rate=0.1)

        print('Epoch %d / %d' % (epoch, args.epochs))

        # SAVE
        if epoch != 1 and epoch % args.save_epoch == 0:
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
            val_score_accum_dict = {}
            val_img_idx = 0
            for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(ValImgLoader):

                vis, scores_list, err_map_list = val_step(model, imgL_crop, imgR_crop, disp_crop_L, args.maxdisp,
                                                          args.testres)

                for score, err_map in zip(scores_list, err_map_list):
                    for (score_tag, score_val), (map_tag, map_val) in zip(score.items(), err_map.items()):
                        log.scalar_summary("val/im_" + str(val_img_idx) + "/" + score_tag, score_val, val_sample_count)
                        log.image_summary("val/" + map_tag, map_val, val_sample_count)

                        if score_tag not in val_score_accum_dict.keys():
                            val_score_accum_dict[score_tag] = 0
                        val_score_accum_dict[score_tag]+=score_val
                    val_img_idx+=1
                    val_sample_count += 1

                log.image_summary('val/left', imgL_crop[0:1], val_sample_count)
                log.image_summary('val/right', imgR_crop[0:1], val_sample_count)
                log.disp_summary('val/gt0', disp_crop_L[0:1], val_sample_count)  # <-- GT disp
                log.entp_summary('val/entropy', vis['entropy'], val_sample_count)
                log.disp_summary('val/output3', vis['output3'][0], val_sample_count)

            for score_tag, score_val in val_score_accum_dict.items():
                log.scalar_summary("val/" + score_tag + "_batch_avg", score_val, epoch)

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            print("training at epoch: " + str(epoch))

            is_scoring = total_iters % 10 == 0

            loss, vis, scores_list, maps = train_step(model, optimizer, imgL_crop, imgR_crop, disp_crop_L, args.maxdisp,
                                                      is_scoring=is_scoring)

            total_train_loss += loss

            if is_scoring:
                log.scalar_summary('train/loss_batch', loss, total_iters)
                for score in scores_list:
                    for tag, val in score.items():
                        log.scalar_summary("train/" + tag + "_batch", val, total_iters)

                        if tag not in train_score_accum_dict.keys():
                            train_score_accum_dict[tag] = 0
                        train_score_accum_dict[tag] += val
                        train_score_accum_dict["num_scored"] += imgL_crop.shape[0]

                for tag, err_map in maps[0].items():
                    log.image_summary("train/" + tag, err_map, total_iters)

            if total_iters % 10 == 0:
                log.image_summary('train/left', imgL_crop[0:1], total_iters)
                log.image_summary('train/right', imgR_crop[0:1], total_iters)
                log.disp_summary('train/gt0', disp_crop_L[0:1], total_iters)  # <-- GT disp
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
