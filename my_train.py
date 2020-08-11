from __future__ import print_function
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
from utils import wandb_logger
from sync_batchnorm.sync_batchnorm import convert_model
from utils import get_metrics
from collections import Counter

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='HSM-Net')
parser.add_argument('--maxdisp', type=int, default=768,
                    help='maxium disparity')
parser.add_argument('--logname', default='logname',
                    help='log name')
parser.add_argument('--database', default='/DATA1/isaac',
                    help='data path')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--batchsize', type=int, default=28,
                    help='samples per batch')
parser.add_argument('--loadmodel', default=None,
                    help='weights path')
parser.add_argument('--savemodel', default='/home/isaac/high-res-stereo/train_output',
                    help='save path')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--val_epoch', type=int, default=1)
parser.add_argument("--sync_bn", action="store_true", default=False)
parser.add_argument("--val", action="store_true", default=False)

args = parser.parse_args()
torch.manual_seed(args.seed)

model = hsm(args.maxdisp, clean=False, level=1)

model = nn.DataParallel(model, device_ids=[0, 1])

if args.sync_bn:
    model = convert_model(model)

model.cuda()

# load model
if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
    pretrained_dict['state_dict'] = {k: v for k, v in pretrained_dict['state_dict'].items() if ('disp' not in k)}
    model.load_state_dict(pretrained_dict['state_dict'], strict=False)

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))


def _init_fn(worker_id):
    np.random.seed()
    random.seed()


torch.manual_seed(args.seed)  # set again
torch.cuda.manual_seed(args.seed)

from dataloader import listfiles as ls
from dataloader import listsceneflow as lt
from dataloader import KITTIloader2015 as lk15
from dataloader import KITTIloader2012 as lk12
from dataloader import MiddleburyLoader as DA

batch_size = args.batchsize
scale_factor = args.maxdisp / 384.  # controls training resolution

# * Gengshan told me to set the scale s.t. the mean of the scale would be same as testres (kitti = 1.8, eth = 3.5 (?) , MB = 1)
kitti_scale_range = [1.4 , 2.2 * scale_factor]  # ? multiply scale_factor or not? Since default maxdisp is 384, scale_factor is 1 by default. (Asked gengshan via FB messenger)
eth_scale_range = [2.7, 4.2 * scale_factor]

all_left_img, all_right_img, all_left_disp, all_right_disp, left_val, right_val, disp_val_L, disp_val_R = ls.hr_dataloader('%s/HR-VS/trainingF' % args.database, val=args.val)
loader_carla = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, right_disparity=all_right_disp,
                                rand_scale=[0.225, 0.6 * scale_factor], rand_bright=[0.8, 1.2], order=2)

val_loader_carla = DA.myImageFloder(left_val, right_val, disp_val_L, right_disparity=disp_val_R,
                                rand_scale=[0.225, 0.6 * scale_factor], rand_bright=[0.8, 1.2], order=2)

all_left_img, all_right_img, all_left_disp, all_right_disp, left_val, right_val, disp_val_L, disp_val_R = ls.mb_dataloader(
    '%s/Middlebury/mb-ex-training/trainingF' % args.database, val=args.val)  # mb-ex
loader_mb = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, right_disparity=all_right_disp,
                             rand_scale=[0.225, 0.6 * scale_factor], rand_bright=[0.8, 1.2], order=0)
val_loader_mb = DA.myImageFloder(left_val, right_val, disp_val_L, right_disparity=disp_val_R,
                             rand_scale=[0.225, 0.6 * scale_factor], rand_bright=[0.8, 1.2], order=0)

all_left_img, all_right_img, all_left_disp, all_right_disp, val_left_img, val_right_img, val_left_disp, val_right_disp = lt.scene_dataloader('%s/SceneFlow/' % args.database, val=args.val)
loader_scene = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, right_disparity=all_right_disp,
                                rand_scale=[0.9, 2.4 * scale_factor], order=2)
val_loader_scene = DA.myImageFloder(val_left_img, val_right_img, val_left_disp, right_disparity=val_right_disp,
                                rand_scale=[0.9, 2.4 * scale_factor], order=2)

all_left_img, all_right_img, all_left_disp, left_val, right_val, disp_val_L = lk15.dataloader('%s/KITTI2015/data_scene_flow/training/' % args.database,
                                                                      val=args.val)  # change to trainval when finetuning on KITTI
loader_kitti15 = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, rand_scale=kitti_scale_range,
                                  order=0)
val_loader_kitti15 = DA.myImageFloder(left_val, right_val, disp_val_L, rand_scale=kitti_scale_range,
                                  order=0)

all_left_img, all_right_img, all_left_disp, left_val, right_val, disp_val_L = lk12.dataloader('%s/KITTI2012/data_stereo_flow/training/' % args.database, val=args.val)
loader_kitti12 = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, rand_scale=kitti_scale_range,
                                  order=0)
val_loader_kitti12 = DA.myImageFloder(left_val, right_val, disp_val_L, rand_scale=kitti_scale_range,
                                  order=0)

# ! Gengshan didn't tell me to change ETH explicitly, just change kitti. Although he didn't mention explicitly, it may be better to chagne ETH as well, so will go a head
all_left_img, all_right_img, all_left_disp, left_val, right_val, disp_val_L = ls.eth_dataloader('%s/ETH3D/low-res-stereo/train/' % args.database, val=args.val)
loader_eth3d = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, rand_scale=kitti_scale_range, order=0)
val_loader_eth3d = DA.myImageFloder(left_val, right_val, disp_val_L, rand_scale=kitti_scale_range, order=0)

# * Gengshan told me to double the loader_kitti15's proportion by multiplying 2
data_inuse = torch.utils.data.ConcatDataset(
    [loader_carla] * 40 + [loader_mb] * 500 + [loader_scene] + [loader_kitti15] * 8 + [loader_kitti12] * 80 + [
        loader_eth3d] * 1000)

data_val = torch.utils.data.ConcatDataset([val_loader_carla, val_loader_eth3d, val_loader_kitti12, val_loader_kitti15, val_loader_mb, val_loader_scene])

TrainImgLoader = torch.utils.data.DataLoader(
    data_inuse,
    batch_size=batch_size, shuffle=True, num_workers=batch_size, drop_last=True, worker_init_fn=_init_fn)

# ValImgLoader = torch.utils.data.DataLoader(
#     data_val, batch_size=batch_size, shuffle=False, num_workers=batch_size, drop_last=False, worker_init_fn=_init_fn)

print('%d batches per epoch' % (len(data_inuse) // batch_size))


def train(imgL, imgR, disp_L):
    model.train()
    imgL = torch.FloatTensor(imgL)
    imgR = torch.FloatTensor(imgR)
    disp_L = torch.FloatTensor(disp_L)

    imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    # ---------
    mask = (disp_true > 0) & (disp_true < args.maxdisp)
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

    del stacked
    del loss
    return lossvalue, vis

def validate(imgL, imgR, disp_L):
    model.eval()

    with torch.no_grad():
        imgL = torch.FloatTensor(imgL)
        imgR = torch.FloatTensor(imgR)
        disp_L = torch.FloatTensor(disp_L)

        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

        # ---------
        mask = (disp_true > 0) & (disp_true < args.maxdisp)
        mask.detach_()
        # ----

        stacked, entropy = model(imgL, imgR)
        loss = (64. / 85) * F.smooth_l1_loss(stacked[0][mask], disp_true[mask], size_average=True) + \
            (16. / 85) * F.smooth_l1_loss(stacked[1][mask], disp_true[mask], size_average=True) + \
            (4. / 85) * F.smooth_l1_loss(stacked[2][mask], disp_true[mask], size_average=True) + \
            (1. / 85) * F.smooth_l1_loss(stacked[3][mask], disp_true[mask], size_average=True)

        vis = {}
        vis['output3'] = stacked[0].detach().cpu().numpy()
        vis['output4'] = stacked[1].detach().cpu().numpy()
        vis['output5'] = stacked[2].detach().cpu().numpy()
        vis['output6'] = stacked[3].detach().cpu().numpy()
        vis['entropy'] = entropy.detach().cpu().numpy()
        lossvalue = loss.data

        del stacked
        del loss
        return lossvalue, vis


def adjust_learning_rate(optimizer, epoch):
    if epoch <= args.epochs - 1:
        lr = 1e-3
    else:
        lr = 1e-4
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    log = wandb_logger.WandbLogger(args)
    log.watch(model)
    total_iters = 0
    save_path = os.path.join(args.savemodel, args.logname + "_" + time.strftime('%l:%M%p_%Y%b%d').strip(" "))
    os.mkdir(save_path)

    for epoch in range(1, args.epochs + 1):
        total_train_loss = 0
        adjust_learning_rate(optimizer, epoch)

        # if epoch % args.val_epoch == 0:
        #     total_val_loss = 0
        #     for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(ValImgLoader):
        #         start_time = time.time()
        #         loss, vis = validate(imgL_crop, imgR_crop, disp_crop_L)
        #         print('Val iter %d training loss = %.3f , time = %.2f' % (batch_idx, loss, time.time() - start_time))
        #         total_val_loss+= loss

        #         if batch_idx % 10 == 0:
        #             log.scalar_summary('val/loss_batch', loss, total_iters)
        #         if batch_idx % 100 == 0:
        #             log.image_summary('val/left', imgL_crop[0:1], total_iters)
        #             log.image_summary('val/right', imgR_crop[0:1], total_iters)
        #             log.image_summary('val/gt0', disp_crop_L[0:1], total_iters) # <-- GT disp
        #             log.image_summary('val/entropy', vis['entropy'][0:1], total_iters)

        #             # ! ERROR: maximum histogram length 512
        #             # log.histo_summary('train/disparity_hist', vis['output3'], total_iters)
        #             # log.histo_summary('train/gt_hist', np.asarray(disp_crop_L), total_iters)

        #             # log outputs of model
        #             log.image_summary('val/output3', vis['output3'][0:1], total_iters)
        #             log.image_summary('val/output4', vis['output4'][0:1], total_iters)
        #             log.image_summary('val/output5', vis['output5'][0:1], total_iters)
        #             log.image_summary('val/output6', vis['output6'][0:1], total_iters)
        #             log.diff_summary("val/output3_diff", disp_crop_L[0:1], vis['output3'][0:1])

        #     log.scalar_summary('val/loss', total_val_loss / len(ValImgLoader), 1)

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time()
            loss, vis = train(imgL_crop, imgR_crop, disp_crop_L)
            print('Epoch %d Iter %d training loss = %.3f , time = %.2f' % (epoch, batch_idx, loss, time.time() - start_time))
            total_train_loss += loss

            if total_iters % 10 == 0:
                log.scalar_summary('train/loss_batch', loss, total_iters)
            if total_iters % 100 == 0:
                # get_metrics(gt, disp, max_disp, datatype, save_path, disp_path=None, gt_path=None, obj_map_path=None, debug=False, ABS_THRESH=3.0, REL_THRESH=0.05)
                # scores = get_metrics.get_metrics()

                log.image_summary('train/left', imgL_crop[0:1], total_iters)
                log.image_summary('train/right', imgR_crop[0:1], total_iters)
                log.image_summary('train/gt0', disp_crop_L[0:1], total_iters, get_color=True) # <-- GT disp
                log.image_summary('train/entropy', vis['entropy'][0:1], total_iters, get_color=True)

                # ! ERROR: maximum histogram length 512
                # log.histo_summary('train/disparity_hist', vis['output3'], total_iters)
                # log.histo_summary('train/gt_hist', np.asarray(disp_crop_L), total_iters)

                # log outputs of model
                log.image_summary('train/output3', vis['output3'][0:1], total_iters, get_color=True)
                log.image_summary('train/output4', vis['output4'][0:1], total_iters, get_color=True)
                log.image_summary('train/output5', vis['output5'][0:1], total_iters, get_color=True)
                log.image_summary('train/output6', vis['output6'][0:1], total_iters, get_color=True)

                # log.diff_summary("train/output3_diff", disp_crop_L[0:1], vis['output3'][0:1], total_iters)

            total_iters += 1

            if (total_iters + 1) % 2000 == 0:
                # SAVE
                savefilename = os.path.join(save_path, 'finetune_' + str(total_iters) + '.tar')

                torch.save({
                    'iters': total_iters,
                    'state_dict': model.state_dict(),
                    'train_loss': total_train_loss / len(TrainImgLoader),
                }, savefilename)

        log.scalar_summary('train/loss', total_train_loss / len(TrainImgLoader), epoch)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
