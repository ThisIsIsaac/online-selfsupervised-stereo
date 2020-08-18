import argparse
import cv2
from models import hsm
import numpy as np
import os
import pdb
import skimage.io
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import time
from models.submodule import *
from utils.eval import mkdir_p, save_pfm
from utils.preprocess import get_transform
from dataloader import KITTIloader2015 as lk15
import subprocess
# cudnn.benchmark = True
cudnn.benchmark = False
import wandb
from dataloader import listfiles as ls
from utils.get_metrics import get_metrics
from statistics import mean
from utils.readpfm import readPFM

parser = argparse.ArgumentParser(description='HSM')
parser.add_argument('--datapath', default='./data-mbtest/',
                    help='test data path')
parser.add_argument('--loadmodel', default=None,
                    help='model path')
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--clean', type=float, default=-1,
                    help='clean up output using entropy estimation')
parser.add_argument('--testres', type=float, default=1,  # Too low for images. Sometimes turn it to 2 ~ 3
                    # for ETH3D we need to use different resolution
                    # 1 - nothibg, 0,5 halves the image, 2 doubles the size of the iamge. We need to
                    # middleburry 1 (3000, 3000)
                    # ETH (3~4) since (1000, 1000)
                    help='test time resolution ratio 0-x')
parser.add_argument('--max_disp', type=float, default=768,
                    help='maximum disparity to search for')
parser.add_argument('--level', type=int, default=1,
                    help='output level of output, default is level 1 (stage 3),\
                          can also use level 2 (stage 2) or level 3 (stage 1)')
args = parser.parse_args()

wandb_logger = wandb.init(name="mb_submission", project="rvc_stereo", save_code=True, magic=True, config=args)

# dataloader
from dataloader import listfiles as DA

# construct model
model = hsm(args.max_disp, args.clean, level=args.level)
model = nn.DataParallel(model, device_ids=[0])
model.cuda()

score_avg = {}

if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
    pretrained_dict['state_dict'] = {k: v for k, v in pretrained_dict['state_dict'].items() if 'disp' not in k}
    model.load_state_dict(pretrained_dict['state_dict'], strict=False)
else:
    print('run with random init')
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

# dry run
multip = 48
imgL = np.zeros((1, 3, 24 * multip, 32 * multip))
imgR = np.zeros((1, 3, 24 * multip, 32 * multip))
imgL = Variable(torch.FloatTensor(imgL).cuda())
imgR = Variable(torch.FloatTensor(imgR).cuda())
with torch.no_grad():
    model.eval()
    pred_disp, entropy = model(imgL, imgR)

_, _, _, _, left_val, right_val, disp_val_L, _ = ls.mb_dataloader('/DATA1/isaac/Middlebury/mb-ex-training/trainingF/',
                                                                      val=True)

def main():
    processed = get_transform()
    model.eval()

    # save predictions
    out_path = os.path.join("./mb_submission_output", args.name)
    if os.path.exists(out_path):
        raise FileExistsError
    os.mkdir(out_path)

    for (left_img_path, right_img_path, disp_path) in zip(left_val, right_val, disp_val_L):
        print(left_img_path)
        imgL_o = (skimage.io.imread(left_img_path).astype('float32'))[:, :, :3]
        imgR_o = (skimage.io.imread(right_img_path).astype('float32'))[:, :, :3]
        gt_o = readPFM(disp_path)[0]

        imgsize = imgL_o.shape[:2]

        with open(os.path.join(left_img_path[:-7], 'calib.txt')) as f:
            lines = f.readlines()
            max_disp = int(int(lines[6].split('=')[-1]))

        ## change max disp
        tmpdisp = int(max_disp * args.testres // 64 * 64)
        if (max_disp * args.testres / 64 * 64) > tmpdisp:
            model.module.maxdisp = tmpdisp + 64
        else:
            model.module.maxdisp = tmpdisp
        if model.module.maxdisp == 64: model.module.maxdisp = 128
        model.module.disp_reg8 = disparityregression(model.module.maxdisp, 16).cuda()
        model.module.disp_reg16 = disparityregression(model.module.maxdisp, 16).cuda()
        model.module.disp_reg32 = disparityregression(model.module.maxdisp, 32).cuda()
        model.module.disp_reg64 = disparityregression(model.module.maxdisp, 64).cuda()

        # resize
        imgL_o = cv2.resize(imgL_o, None, fx=args.testres, fy=args.testres, interpolation=cv2.INTER_CUBIC)
        imgR_o = cv2.resize(imgR_o, None, fx=args.testres, fy=args.testres, interpolation=cv2.INTER_CUBIC)
        imgL = processed(imgL_o).numpy()
        imgR = processed(imgR_o).numpy()

        imgL = np.reshape(imgL, [1, 3, imgL.shape[1], imgL.shape[2]])
        imgR = np.reshape(imgR, [1, 3, imgR.shape[1], imgR.shape[2]])

        ##fast pad
        max_h = int(imgL.shape[2] // 64 * 64)
        max_w = int(imgL.shape[3] // 64 * 64)
        if max_h < imgL.shape[2]: max_h += 64
        if max_w < imgL.shape[3]: max_w += 64

        top_pad = max_h - imgL.shape[2]
        left_pad = max_w - imgL.shape[3]
        imgL = np.lib.pad(imgL, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)
        imgR = np.lib.pad(imgR, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)



        # test
        imgL = torch.FloatTensor(imgL)
        imgR = torch.FloatTensor(imgR)
        wandb.log(
            {"imgL": wandb.Image(imgL, caption=str(imgL.shape)), "imgR": wandb.Image(imgR, caption=str(imgR.shape))})
        imgL = imgL.cuda()
        imgR = imgR.cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            pred_disp, entropy = model(imgL, imgR)
            torch.cuda.synchronize()
            ttime = (time.time() - start_time)

        pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()

        top_pad = max_h - imgL_o.shape[0]
        left_pad = max_w - imgL_o.shape[1]
        entropy = entropy[top_pad:, :pred_disp.shape[1] - left_pad].cpu().numpy()
        pred_disp = pred_disp[top_pad:, :pred_disp.shape[1] - left_pad]

        # save predictions
        idxname = left_img_path.split('/')[-2]
        if not os.path.exists('%s/%s'%(out_path,idxname)):
            os.makedirs('%s/%s'%(out_path,idxname))
        idxname = '%s/disp0HSM'%(idxname)

        with open('%s/%s.pfm'% (out_path, idxname),'w') as f:
            save_pfm(f,pred_disp[::-1,:])
        with open('%s/%s/timeHSM.txt'%(out_path,idxname.split('/')[0]),'w') as f:
             f.write(str(ttime))

        # resize to highres
        pred_disp = cv2.resize(pred_disp / args.testres, (imgsize[1], imgsize[0]), interpolation=cv2.INTER_LINEAR)

        # clip while keep inf
        invalid = np.logical_or(pred_disp == np.inf, pred_disp != pred_disp)
        pred_disp[invalid] = np.inf

        pred_disp_png = (pred_disp * 256).astype('uint16')
        cv2.imwrite(os.path.join(out_path, idxname.split('/')[0] + ".png"), pred_disp_png)
        entropy_png = (entropy * 256).astype('uint16')
        # cv2.imwrite(os.path.join(out_dir, img_name), entropy_png)

        wandb.log({"disp": wandb.Image(pred_disp_png, caption=str(pred_disp_png.shape)),
                   "entropy": wandb.Image(entropy_png, caption=str(entropy_png.shape))})

        metrics = get_metrics(gt_o, pred_disp, max_disp)

        for (key, val) in metrics.items():
            if key not in score_avg.keys():
                score_avg[key] = []
            score_avg[key].append(val)

        torch.cuda.empty_cache()

    for (key, val) in score_avg.items():
        score_avg[key] = mean(score_avg[key])

    print(score_avg)
    with open(os.path.join(out_path, "metrrics.txt")) as file:
        file.write(str(score_avg))
    
if __name__ == '__main__':
    main()

