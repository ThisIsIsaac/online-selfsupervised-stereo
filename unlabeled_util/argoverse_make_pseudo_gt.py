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
from datetime import datetime
# cudnn.benchmark = True
cudnn.benchmark = False
import math



def main():
    parser = argparse.ArgumentParser(description='HSM')
    parser.add_argument("--name", required=True)
    parser.add_argument('--datapath', default='./data-mbtest/',
                        help='test data path')
    parser.add_argument('--loadmodel', default=None,
                        help='model path')
    parser.add_argument('--clean', type=float, default=-1,
                        help='clean up output using entropy estimation')
    parser.add_argument('--testres', type=float, default=1.8,  # Too low for images. Sometimes turn it to 2 ~ 3
                        # for ETH3D we need to use different resolution
                        # 1 - nothibg, 0,5 halves the image, 2 doubles the size of the iamge. We need to
                        # middleburry 1 (3000, 3000)
                        # ETH (3~4) since (1000, 1000)
                        help='test time resolution ratio 0-x')
    parser.add_argument('--max_disp', type=int, default=2056,
                        help='maximum disparity to search for')
    parser.add_argument('--level', type=int, default=1,
                        help='output level of output, default is level 1 (stage 3),\
                            can also use level 2 (stage 2) or level 3 (stage 1)')
    args = parser.parse_args()
    args.max_disp = int(args.max_disp*args.testres) # max_disp = 2056 * testres
    args.max_disp = 16 * math.floor(args.max_disp/16)

    args.name = args.name + "_"+ datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    # dataloader
    from dataloader import listfiles as DA

    # test_left_img, test_right_img, _, _ = DA.dataloader(args.datapath)
    # print("total test images: " + str(len(test_left_img)))
    # print("output path: " + args.outdir)

    # construct model
    model = hsm(args.max_disp, args.clean, level=args.level)
    model = nn.DataParallel(model, device_ids=[0])
    model.cuda()

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

    left_img_dir = os.path.join(args.datapath, "stereo_front_left")
    right_img_dir = os.path.join(args.datapath, "stereo_front_right")

    left_img_path_list = os.listdir(left_img_dir)
    left_img_path_list.sort()
    right_img_path_list = os.listdir(right_img_dir)
    right_img_path_list.sort()

    processed = get_transform()
    model.eval()

    # save predictions
    out_path = os.path.join(args.datapath, args.name)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    disp_path = os.path.join(out_path, "disp")
    entp_path = os.path.join(out_path, "entropy")

    if not os.path.exists(disp_path):
        os.mkdir(disp_path)

    if not os.path.exists(entp_path):
        os.mkdir(entp_path)
    

    
    for (left_img_name, right_img_name) in zip(left_img_path_list, right_img_path_list):

        left_img_path = os.path.join(left_img_dir, left_img_name)
        right_img_path = os.path.join(right_img_dir, right_img_name)

        print(left_img_path)

        imgL_o = (skimage.io.imread(left_img_path).astype('float32'))[:, :, :3]
        imgR_o = (skimage.io.imread(right_img_path).astype('float32'))[:, :, :3]
        imgsize = imgL_o.shape[:2]

        max_disp = int(args.max_disp)

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

        imgL = imgL.cuda()
        imgR = imgR.cuda()

        with torch.no_grad():
            torch.cuda.synchronize()

            pred_disp, entropy = model(imgL, imgR)
            torch.cuda.synchronize()

        pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()

        top_pad = max_h - imgL_o.shape[0]
        left_pad = max_w - imgL_o.shape[1]
        entropy = entropy[top_pad:, :pred_disp.shape[1] - left_pad].cpu().numpy()
        pred_disp = pred_disp[top_pad:, :pred_disp.shape[1] - left_pad]

        # resize to highres
        pred_disp = cv2.resize(pred_disp / args.testres, (imgsize[1], imgsize[0]), interpolation=cv2.INTER_LINEAR)

        # clip while keep inf
        invalid = np.logical_or(pred_disp == np.inf, pred_disp != pred_disp)
        pred_disp[invalid] = np.inf

        out_file_name = left_img_path[len(left_img_path) - len("315970554564438888.jpg"): len(left_img_path) - len("jpg")]
        out_file_name = os.path.join(out_file_name + "png", )

        pred_disp_png = (pred_disp * 256).astype('uint16')
        cv2.imwrite(os.path.join(disp_path, out_file_name), pred_disp_png)
        # entropy_png = (entropy* 256).astype('uint16')
        entropy_png = (entropy/entropy.max()*256)
        cv2.imwrite(os.path.join(entp_path, out_file_name), entropy_png)

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()

