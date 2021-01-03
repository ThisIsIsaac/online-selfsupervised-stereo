from rich import print
from rich import pretty
pretty.install()
from rich import traceback
traceback.install()
import random
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
from utils import kitti_eval
import math
torch.backends.cudnn.benchmark = True
import torch.nn.functional as F

def _init_fn(worker_id):
    np.random.seed()
    random.seed()

def train_step(model, optimizer, imgL, imgR, disp_L, maxdisp, gpu=0, is_scoring=False):
    model.train()
    imgL = torch.FloatTensor(imgL)
    imgR = torch.FloatTensor(imgR)
    disp_L = torch.FloatTensor(disp_L)

    imgL, imgR, disp_true = imgL.cuda(gpu), imgR.cuda(gpu), disp_L.cuda(gpu)

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
    eval_maps= None
    if is_scoring:
        eval_score, eval_maps = kitti_eval.evaluate(disp_true.detach().cpu().numpy(), vis['output3'][0])

    del stacked
    del loss
    return lossvalue, vis, eval_score, eval_maps

def val_step(model, imgL, imgR, disp_L, maxdisp, testres, gpu=0):
    model.eval()
    with torch.no_grad():
        imgL = torch.FloatTensor(imgL)
        imgR = torch.FloatTensor(imgR)
        disp_L = torch.FloatTensor(disp_L)

        imgL, imgR, disp_true = imgL.cuda(gpu), imgR.cuda(gpu), disp_L.cuda(gpu)

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

        vis = {}
        vis['output3'] = out.detach().cpu().numpy()
        vis['entropy'] = entropy.detach().cpu().numpy()
        eval_score, eval_maps = kitti_eval.evaluate(disp_true.detach().cpu().numpy(), vis['output3'][0])

        return vis, eval_score, eval_maps


def adjust_learning_rate(optimizer, init_lr, decay_epoch, epoch, total_epochs, decay_rate=0.1):
    # * ratio of lr to batch_size is 0.01:28
    # lr = 1e-3 * (batch_size / 28)
    if epoch > total_epochs - decay_epoch:
        lr = init_lr * decay_rate
    else:
        lr = init_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
