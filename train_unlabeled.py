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
from utils import wandb_logger, kitti_eval
from sync_batchnorm.sync_batchnorm import convert_model
from utils import get_metrics
from collections import Counter


def unlabeled_loader():
    disp_dir = "/home/isaac/high-res-stereo/unlabeled_util/pseudo_gt/disp/"
    entropy_dir = "/home/isaac/high-res-stereo/unlabeled_util/pseudo_gt/entropy/"

    disp_paths = []
    entropy_paths = []

    for img in os.listdir(disp_dir):
        disp_paths.append(os.path.join(disp_dir, img))
        entropy_paths.append(os.path.join(entropy_dir, img))

    left_img_paths = []
    right_img_paths = []

    for file in disp_paths:
        name = file.split("/")[-1]
        name = name.replace("-", "/")
        name = os.path.join("/DATA1/isaac/KITTI_raw/", name)
        left_img_paths.append(name)
        right_img_paths.append(name.replace("/image_02/", "/image_03/"))

    return left_img_paths, right_img_paths, disp_paths, entropy_paths


torch.backends.cudnn.benchmark = True

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
parser.add_argument('--batchsize', type=int, default=18, # when maxdisp is 768, 18 is the most you can fit in 2 V100s (with syncBN on)
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
parser.add_argument('--threshold', type=float, default=None, help='entropy threshold')

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

# initial lr will be set in "adjust_learning_rate" function. this is just a dummy variable for init
dummy_lr = 0.1
optimizer = optim.Adam(model.parameters(), lr=dummy_lr, betas=(0.9, 0.999))


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
kitti_scale_range = [1.4 , 2.2]  # ? multiply scale_factor or not? Since default maxdisp is 384, scale_factor is 1 by default. (Asked gengshan via FB messenger)

all_left_img, all_right_img, all_left_disp, all_left_entropy = unlabeled_loader()


loader = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, left_entropy=all_left_entropy, rand_scale=kitti_scale_range,
                                  order=0, entropy_threshold=args.threshold) # or 0.95


TrainImgLoader = torch.utils.data.DataLoader(
    loader,
    batch_size=batch_size, shuffle=False, num_workers=batch_size, drop_last=True, worker_init_fn=_init_fn)

print('%d batches per epoch' % (len(loader) // batch_size))


def train(imgL, imgR, disp_L):
    model.train()
    imgL = torch.FloatTensor(imgL)
    imgR = torch.FloatTensor(imgR)
    disp_L = torch.FloatTensor(disp_L)

    imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    # ---------
    mask = (disp_true > 0) & (disp_true < args.maxdisp) # 
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

    errors = kitti_eval.evaluate(disp_L, stacked[0])

    del stacked
    del loss
    return lossvalue, vis, errors

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

        errors = kitti_eval.evaluate(disp_L, stacked[0])

        del stacked
        del loss
        return lossvalue, vis, errors


def adjust_learning_rate(optimizer, epoch):
    # * ratio of lr to batchsize is 0.01:28 
    lr = 1e-3 * (args.batchsize / 28)
    if epoch > args.epochs - 1:
        lr = lr / 10

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

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time()
            loss, vis, errors = train(imgL_crop, imgR_crop, disp_crop_L)
            print('Epoch %d Iter %d training loss = %.3f , time = %.2f' % (epoch, batch_idx, loss, time.time() - start_time))
            total_train_loss += loss

            if total_iters % 10 == 0:
                # result = subprocess.run(["/home/isaac/KITTI2015_devkit/cpp/eval_stereo_flow", 
                #                          out_path+"/"], stdout=subprocess.PIPE)
                
                # print(result.stdout)

                log.scalar_summary('train/loss_batch', loss, total_iters)

            if total_iters % 100 == 0:
                
                log.image_summary('train/left', imgL_crop[0:1], total_iters)
                log.image_summary('train/right', imgR_crop[0:1], total_iters)
                log.image_summary('train/gt0', disp_crop_L[0:1], total_iters, get_color=True) # <-- GT disp
                log.image_summary('train/entropy', vis['entropy'][0:1], total_iters, get_color=True)

                # log outputs of model
                log.image_summary('train/output3', vis['output3'][0:1], total_iters, get_color=True)
                log.image_summary('train/output4', vis['output4'][0:1], total_iters, get_color=True)
                log.image_summary('train/output5', vis['output5'][0:1], total_iters, get_color=True)
                log.image_summary('train/output6', vis['output6'][0:1], total_iters, get_color=True)

            total_iters += 1

            if (total_iters + 1) % 2000 == 0:
                # SAVE
                savefilename = os.path.join(save_path, 'finetune_' + str(total_iters) + '.tar')

                torch.save({
                    'iters': total_iters,
                    'state_dict': model.state_dict(),
                    'train_loss': total_train_loss / len(TrainImgLoader),
                }, savefilename)

        if args.val and epoch % args.val_epoch == 0:
            for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(ValImgLoader):
                loss, vis, errors = validate(imgL_crop, imgR_crop, disp_crop_L)

                log.image_summary('val/left', imgL_crop[0:1], total_iters)
                log.image_summary('val/right', imgR_crop[0:1], total_iters)
                log.image_summary('val/gt0', disp_crop_L[0:1], total_iters, get_color=True) # <-- GT disp
                log.image_summary('val/entropy', vis['entropy'][0:1], total_iters, get_color=True)

                # log outputs of model
                log.image_summary('val/output3', vis['output3'][0:1], total_iters, get_color=True)
                log.image_summary('val/output4', vis['output4'][0:1], total_iters, get_color=True)
                log.image_summary('val/output5', vis['output5'][0:1], total_iters, get_color=True)
                log.image_summary('val/output6', vis['output6'][0:1], total_iters, get_color=True)
                log.scalar_summary('val/loss_batch', loss, total_iters)

        log.scalar_summary('val/loss', total_train_loss / len(TrainImgLoader), epoch)
        log.scalar_summary("val/errors", )
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
