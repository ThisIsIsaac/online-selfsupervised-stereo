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

parser = argparse.ArgumentParser(description='HSM')
parser.add_argument('--datapath', default='./data-mbtest/',
                    help='test data path')
parser.add_argument('--loadmodel', default=None,
                    help='model path')
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--clean', type=float, default=-1,
                    help='clean up output using entropy estimation')
parser.add_argument('--testres', type=float, default=1.8,  # Too low for images. Sometimes turn it to 2 ~ 3
                    # for ETH3D we need to use different resolution
                    # 1 - nothibg, 0,5 halves the image, 2 doubles the size of the iamge. We need to
                    # middleburry 1 (3000, 3000)
                    # ETH (3~4) since (1000, 1000)
                    help='test time resolution ratio 0-x')
parser.add_argument('--max_disp', type=float, default=384,
                    help='maximum disparity to search for')
parser.add_argument('--level', type=int, default=1,
                    help='output level of output, default is level 1 (stage 3),\
                          can also use level 2 (stage 2) or level 3 (stage 1)')
args = parser.parse_args()

wandb_logger = wandb.init(name="submission.py", project="rvc_stereo", save_code=True, magic=True, config=args)

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

_, _, _, left_val, right_val, disp_val_L = lk15.dataloader('/DATA1/isaac/KITTI2015/data_scene_flow/training/',
                                                                      val=True)

def main():
    processed = get_transform()
    model.eval()

    # save predictions
    out_path = os.path.join("./kitti_submission_output", args.name)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    out_dir = os.path.join(out_path, "disp_0")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for (left_img_path, right_img_path, disp_path) in zip(left_val, right_val, disp_val_L):
        # print(test_left_img[inx])
        print(left_img_path)
        imgL_o = (skimage.io.imread(left_img_path).astype('float32'))[:, :, :3]
        imgR_o = (skimage.io.imread(right_img_path).astype('float32'))[:, :, :3]
        imgsize = imgL_o.shape[:2]
        # torch.save(imgL_o, "/home/isaac/high-res-stereo/debug/my_submission/img0.pt")

        if args.max_disp > 0:
            max_disp = int(args.max_disp)
        else:
            with open('/home/isaac/rvc_devkit/stereo/datasets_middlebury2014/training/Kitti2015_000028_10/calib.txt') as f:
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
        # torch.save(imgL_o, "/home/isaac/high-res-stereo/debug/my_submission/img1.pt")

        imgL = processed(imgL_o).numpy()
        imgR = processed(imgR_o).numpy()
        # torch.save(imgL, "/home/isaac/high-res-stereo/debug/my_submission/img2.pt")

        imgL = np.reshape(imgL, [1, 3, imgL.shape[1], imgL.shape[2]])
        imgR = np.reshape(imgR, [1, 3, imgR.shape[1], imgR.shape[2]])
        # torch.save(imgL, "/home/isaac/high-res-stereo/debug/my_submission/img3.pt")

        ##fast pad
        max_h = int(imgL.shape[2] // 64 * 64)
        max_w = int(imgL.shape[3] // 64 * 64)
        if max_h < imgL.shape[2]: max_h += 64
        if max_w < imgL.shape[3]: max_w += 64

        top_pad = max_h - imgL.shape[2]
        left_pad = max_w - imgL.shape[3]
        imgL = np.lib.pad(imgL, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)
        imgR = np.lib.pad(imgR, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)
        # torch.save(imgL, "/home/isaac/high-res-stereo/debug/my_submission/img4.pt")

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

            # torch.save(imgL, "/home/isaac/high-res-stereo/debug/my_submission/img_final.pt")

            pred_disp, entropy = model(imgL, imgR)
            torch.cuda.synchronize()
            ttime = (time.time() - start_time)

        pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()

        top_pad = max_h - imgL_o.shape[0]
        left_pad = max_w - imgL_o.shape[1]
        entropy = entropy[top_pad:, :pred_disp.shape[1] - left_pad].cpu().numpy()
        pred_disp = pred_disp[top_pad:, :pred_disp.shape[1] - left_pad]

        img_name = os.path.basename(os.path.normpath(left_img_path))

        # resize to highres
        pred_disp = cv2.resize(pred_disp / args.testres, (imgsize[1], imgsize[0]), interpolation=cv2.INTER_LINEAR)

        # clip while keep inf
        invalid = np.logical_or(pred_disp == np.inf, pred_disp != pred_disp)
        pred_disp[invalid] = np.inf

        # np.save('%s/%s' % (out_dir, img_name), (pred_disp))
        # np.save('%s/%s-ent.npy' % (out_dir, idxname.split('/')[0]), (entropy))
        pred_disp_png = (pred_disp * 256).astype('uint16')
        cv2.imwrite(os.path.join(out_dir, img_name), pred_disp_png)
        entropy_png = (entropy * 256).astype('uint16')
        # cv2.imwrite(os.path.join(out_dir, img_name), entropy_png)

        wandb.log({"disp": wandb.Image(pred_disp_png, caption=str(pred_disp_png.shape)),
                   "entropy": wandb.Image(entropy_png, caption=str(entropy_png.shape))})

        # with open('%s/%s.pfm' % (out_dir, idxname), 'w') as f:
        #     save_pfm(f, pred_disp[::-1, :])
        # with open('%s/%s/timeHSM.txt' % (out_dir, idxname.split('/')[0]), 'w') as f:
        #     f.write(str(ttime))

        torch.cuda.empty_cache()

    subprocess.run(["/home/isaac/KITTI2015_devkit/cpp/eval_scene_flow", out_path+"/"])



if __name__ == '__main__':
    main()

