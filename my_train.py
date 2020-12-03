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
from sync_batchnorm.sync_batchnorm import convert_model
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


def _init_fn(worker_id):
    np.random.seed()
    random.seed()

def train_step(model, optimizer, imgL, imgR, disp_L, maxdisp, is_scoring=False):
    model.train()
    imgL = torch.FloatTensor(imgL)
    imgR = torch.FloatTensor(imgR)
    disp_L = torch.FloatTensor(disp_L)

    imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    # ---------
    mask = (disp_true > 0) & (disp_true < maxdisp) #
    mask.detach_()
    # ----

    optimizer.zero_grad()
    stacked, entropy = model(imgL, imgR)
    loss = (64. / 85) * F.smooth_l1_loss(stacked[0][mask], disp_true[mask], size_average=True) + \
           (16. / 85) * F.smooth_l1_loss(stacked[1][mask], disp_true[mask], size_average=True) + \
           (4. / 85) * F.smooth_l1_loss(stacked[2][mask], disp_true[mask], size_average=True) + \
           (1. / 85) * F.smooth_l1_loss(stacked[3][mask], disp_true[mask], size_average=True)
    loss.backward()
    optimizer.step()
    vis = {}
    vis['output3'] = stacked[0].detach().cpu().numpy()
    vis['output4'] = stacked[1].detach().cpu().numpy()
    vis['output5'] = stacked[2].detach().cpu().numpy()
    vis['output6'] = stacked[3].detach().cpu().numpy()
    vis['entropy'] = entropy.detach().cpu().numpy()
    lossvalue = loss.data

    eval_score = None
    if is_scoring:
        eval_score = kitti_eval.evaluate(disp_true.detach().cpu().numpy(), vis['output3'][0])

    del stacked
    del loss
    return lossvalue, vis, eval_score

def val_step(model, imgL, imgR, disp_L, maxdisp, testres):
    model.eval()

    with torch.no_grad():
        imgL = torch.FloatTensor(imgL)
        imgR = torch.FloatTensor(imgR)
        disp_L = torch.FloatTensor(disp_L)

        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

        # ---------
        mask = (disp_true > 0) & (disp_true < maxdisp)
        mask.detach_()
        # ----

        out, entropy = model(imgL, imgR)

        top_pad = out.shape[-2] - math.ceil(testres * disp_L.shape[-2])
        left_pad = out.shape[-1] - math.ceil(testres * disp_L.shape[-1])
        out = out[:, top_pad:, :out.shape[-1] - left_pad]
        out = torch.unsqueeze(out, 0)
        out = F.interpolate(out / testres, size=(disp_L.shape[-2], disp_L.shape[-1]), mode="bilinear")

        # loss = (64. / 85) * F.smooth_l1_loss(stacked[0][mask], disp_true[mask], size_average=True) + \
        #     (16. / 85) * F.smooth_l1_loss(stacked[1][mask], disp_true[mask], size_average=True) + \
        #     (4. / 85) * F.smooth_l1_loss(stacked[2][mask], disp_true[mask], size_average=True) + \
        #     (1. / 85) * F.smooth_l1_loss(stacked[3][mask], disp_true[mask], size_average=True)

        vis = {}
        vis['output3'] = out.detach().cpu().numpy()
        # vis['output4'] = stacked[1].detach().cpu().numpy()
        # vis['output5'] = stacked[2].detach().cpu().numpy()
        # vis['output6'] = stacked[3].detach().cpu().numpy()
        vis['entropy'] = entropy.detach().cpu().numpy()
        # lossvalue = loss.data

        eval_score = kitti_eval.evaluate(disp_true.detach().cpu().numpy(), vis['output3'][0])

        # del stacked
        # del loss
        return vis, eval_score


def adjust_learning_rate(batchsize, epochs, optimizer, epoch):
    # * ratio of lr to batchsize is 0.01:28 
    lr = 1e-3 * (batchsize / 28)
    if epoch > epochs - 1:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    parser = argparse.ArgumentParser(description='HSM-Net')
    parser.add_argument('--maxdisp', type=int, default=384,
                        help='maxium disparity')
    parser.add_argument('--name', default='name')
    parser.add_argument('--database', default='/data/private',
                        help='data path')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train')
    parser.add_argument('--batchsize', type=int, default=18,
                        # when maxdisp is 768, 18 is the most you can fit in 2 V100s (with syncBN on)
                        help='samples per batch')
    parser.add_argument('--loadmodel', default=None,
                        help='weights path')
    parser.add_argument('--log_dir', default="/data/private/tensorboard/high-res-stereo")
    parser.add_argument('--savemodel', default=os.path.join(os.getcwd(),'/trained_model'),
                        help='save path')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--val_epoch', type=int, default=10)
    parser.add_argument("--sync_bn", action="store_true", default=False)
    parser.add_argument("--val", action="store_true", default=False)
    parser.add_argument("--save_numpy", action="store_true", default=False)
    parser.add_argument("--testres", type=float, default=1.8)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--use_pseudoGT", default=False, action="store_true")

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.manual_seed(args.seed)  # set again
    torch.cuda.manual_seed(args.seed)
    batch_size = args.batchsize
    scale_factor = args.maxdisp / 384.  # controls training resolution
    args.name = args.name + "_" + time.strftime('%l:%M%p_%Y%b%d').strip(" ")


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
        val=args.val)  # change to trainval when finetuning on KITTI
    # all_left_img = [left_val[3]] * args.batchsize
    # all_right_img = [right_val[3]] * args.batchsize

    all_left_img = []
    all_right_img = []

    left_img_name= os.listdir("/data/private/KITTI_raw/2011_09_26/2011_09_26_drive_0013_sync/image_02/data")
    right_img_name= os.listdir("/data/private/KITTI_raw/2011_09_26/2011_09_26_drive_0013_sync/image_03/data")
    left_disp_path_all = os.listdir("/data/private/KITTI_raw/2011_09_26/2011_09_26_drive_0013_sync/final-768px/disp")
    left_entropy_path_all = os.listdir("/data/private/KITTI_raw/2011_09_26/2011_09_26_drive_0013_sync/final-768px/entropy")

    left_img_name.sort()
    right_img_name.sort()
    left_disp_path_all.sort()
    left_entropy_path_all.sort()

    all_left_disp = []
    left_entropy = []
    for i in range(len(left_disp_path_all)):
        if left_disp_path_all[i].endswith(".npy"):
            p = os.path.join("/data/private/KITTI_raw/2011_09_26/2011_09_26_drive_0013_sync/final-768px/disp", left_disp_path_all[i])
            all_left_disp.append(p)
        if left_entropy_path_all[i].endswith(".npy"):
            p = os.path.join("/data/private/KITTI_raw/2011_09_26/2011_09_26_drive_0013_sync/final-768px/entropy",
                             left_entropy_path_all[i])
            left_entropy.append(p)

    all_left_img = []
    all_right_img = []
    for i in range(len(left_img_name)):
        l_p = os.path.join("/data/private/KITTI_raw/2011_09_26/2011_09_26_drive_0013_sync/image_02/data", left_img_name[i])
        r_p = os.path.join("/data/private/KITTI_raw/2011_09_26/2011_09_26_drive_0013_sync/image_03/data", right_img_name[i])

        all_left_img.append(l_p)
        all_right_img.append(r_p)

    # all_left_disp = ["/data/private/KITTI_raw/2011_09_26/2011_09_26_drive_0013_sync/final-768px/disp/0000000040.npy"] * args.batchsize
    # left_entropy = ["/data/private/KITTI_raw/2011_09_26/2011_09_26_drive_0013_sync/final-768px/entropy/0000000040.npy"] * args.batchsize

    left_val = [left_val[3]]
    right_val = [right_val[3]]
    disp_val_L = [disp_val_L[3]]

    loader_kitti15 = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, rand_scale=[0.9, 2.4 * scale_factor],
                                      order=0, use_pseudoGT=args.use_pseudoGT, entropy_threshold=args.threshold,
                                      left_entropy=left_entropy)
    val_loader_kitti15 = DA.myImageFloder(left_val, right_val, disp_val_L, is_validation=True, testres=1.8)

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

    ValImgLoader = torch.utils.data.DataLoader(val_data_inuse, drop_last=False, worker_init_fn=_init_fn, batch_size=1,
                                               shuffle=False)

    TrainImgLoader = torch.utils.data.DataLoader(
        train_data_inuse,
        batch_size=batch_size, shuffle=True, num_workers=batch_size, drop_last=True, worker_init_fn=_init_fn)

    print('%d batches per epoch' % (len(train_data_inuse) // batch_size))

    model = hsm(args.maxdisp, clean=False, level=1)

    model = nn.DataParallel(model, device_ids=[0])

    if args.sync_bn:
        model = convert_model(model)

    model.cuda()

    # load model
    if args.loadmodel is not None:
        pretrained_dict = torch.load(args.loadmodel)
        pretrained_dict['state_dict'] = {k: v for k, v in pretrained_dict['state_dict'].items() if ('disp' not in k)}
        model.load_state_dict(pretrained_dict['state_dict'], strict=False)

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # initial lr will be set in "adjust_learning_rate" function. this is just a dummy variable for init
    dummy_lr = 0.1
    optimizer = optim.Adam(model.parameters(), lr=dummy_lr, betas=(0.9, 0.999))

    log = logger.Logger(args.log_dir, args.name, save_numpy=args.save_numpy)
    # log.watch(model)
    total_iters = 0
    val_iters = 0


    save_path = os.path.join(args.savemodel, args.name)
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        total_train_loss = 0
        train_score_accum_dict = {} # accumulates scores throughout a batch to get average score
        train_score_accum_dict["num_scored"] = 0
        adjust_learning_rate(args.batchsize, args.epochs, optimizer, epoch)
        print('Epoch %d / %d' % (epoch, args.epochs))

        ## val ##
        if epoch % args.val_epoch == 0:
        # if True:
            print("validating at epoch: " + str(epoch))
            total_val_loss = 0
            val_score_accum_dict = {} # accumulates scores throughout a batch to get average score
            for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(ValImgLoader):


                # start_time = time.time()
                vis, scores_list = val_step(model, imgL_crop, imgR_crop, disp_crop_L, args.maxdisp, args.testres)
                # print('Epoch %d Iter %d training loss = %.3f , time = %.2f' % (
                #        epoch, batch_idx, loss, time.time() - start_time))
                # total_val_loss += loss

                # log.scalar_summary('val/loss_batch', loss, val_iters)

                for score in scores_list:
                    for tag, val in score.items():
                        log.scalar_summary("val/" + tag + "_batch", val, val_iters)

                        if tag not in val_score_accum_dict.keys():
                            val_score_accum_dict[tag] = 0
                        val_score_accum_dict[tag] += val

                log.image_summary('val/left', imgL_crop[0:1], val_iters)
                log.image_summary('val/right', imgR_crop[0:1], val_iters)
                log.disp_summary('val/gt0', disp_crop_L[0:1], val_iters)  # <-- GT disp
                log.entp_summary('val/entropy', vis['entropy'], val_iters)
                log.disp_summary('val/output3', vis['output3'][0], val_iters)
                # log.disp_summary('val/output4', vis['output4'][0:1], val_iters)
                # log.disp_summary('val/output5', vis['output5'][0:1], val_iters)
                # log.disp_summary('val/output6', vis['output6'][0:1], val_iters)

                val_iters += 1

            # log.scalar_summary('val/loss_avg', total_val_loss / len(ValImgLoader), epoch)
            # for tag, val in val_score_accum_dict.items():
            #     log.scalar_summary("val/" + tag + "_avg", val/len(ValImgLoader), epoch)

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):

            is_scoring = total_iters%10 == 0

            loss, vis, scores_list = train_step(model, optimizer, imgL_crop, imgR_crop, disp_crop_L, args.maxdisp, is_scoring=is_scoring)

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

            if total_iters % 10 == 0:
                log.image_summary('train/left', imgL_crop[0:1], total_iters)
                log.image_summary('train/right', imgR_crop[0:1], total_iters)
                log.disp_summary('train/gt0', disp_crop_L[0:1], total_iters) # <-- GT disp
                log.entp_summary('train/entropy', vis['entropy'][0:1], total_iters)
                log.disp_summary('train/output3', vis['output3'][0:1], total_iters)
                # log.disp_summary('train/output4', vis['output4'][0:1], total_iters)
                # log.disp_summary('train/output5', vis['output5'][0:1], total_iters)
                # log.disp_summary('train/output6', vis['output6'][0:1], total_iters)

            total_iters += 1

            if (total_iters + 1) % 2000 == 0:
                # SAVE
                print("saving weights at epoch: " + str(epoch))
                savefilename = os.path.join(save_path, 'finetune_' + str(total_iters) + '.tar')

                torch.save({
                    'iters': total_iters,
                    'state_dict': model.state_dict(),
                    'train_loss': total_train_loss / len(TrainImgLoader),
                }, savefilename)

        log.scalar_summary('train/loss', total_train_loss / len(TrainImgLoader), epoch)
        # for tag, val in train_score_accum_dict.items():
        #     log.scalar_summary("train/" + tag + "_avg", val / train_score_accum_dict["num_scored"], epoch)
        torch.cuda.empty_cache()





if __name__ == '__main__':
    main()
