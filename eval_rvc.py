import argparse
import cv2
from models import hsm
import os
import skimage.io
import torch.backends.cudnn as cudnn
import time
from models.submodule import *
from utils.eval import save_pfm
from utils.preprocess import get_transform
import sys
from dataset_format import *
from dataloader.RVCDataset import RVCDataset
from torch.utils.data import Dataset, DataLoader
# cudnn.benchmark = True
cudnn.benchmark = False
import wandb

# source: `rvc_devkit/stereo/util_stereo.py`
# Returns a dict which maps the parameters to their values. The values (right
# side of the equal sign) are all returned as strings (and not parsed).
def ReadMiddlebury2014CalibFile(path):
    result = dict()
    with open(path, 'rb') as calib_file:
        for line in calib_file.readlines():
            line = line.decode('UTF-8').rstrip('\n')
            if len(line) == 0:
                continue
            eq_pos = line.find('=')
            if eq_pos < 0:
                raise Exception('Cannot parse Middlebury 2014 calib file: ' + path)
            result[line[:eq_pos]] = line[eq_pos + 1:]
    return result

def main():
    parser = argparse.ArgumentParser(description='HSM')
    parser.add_argument('--datapath', default="/home/isaac/rvc_devkit/stereo/datasets_middlebury2014",
                        help='test data path')
    parser.add_argument('--loadmodel', default=None,
                        help='model path')
    parser.add_argument('--name', default='rvc_highres_output',
                        help='output dir')
    parser.add_argument('--clean', type=float, default=-1,
                        help='clean up output using entropy estimation')
    parser.add_argument('--testres', type=float, default=-1, #default used to be 0.5
                        help='test time resolution ratio 0-x')
    parser.add_argument('--max_disp', type=float, default=-1,
                        help='maximum disparity to search for')
    parser.add_argument('--level', type=int, default=1,
                        help='output level of output, default is level 1 (stage 3),\
                              can also use level 2 (stage 2) or level 3 (stage 1)')
    parser.add_argument('--debug_image', type=str, default=None)
    parser.add_argument("--eth_testres" , type=int, default=3.5)
    args = parser.parse_args()

    wandb.init(name=args.name, project="rvc_stereo", save_code=True, magic=True, config=args)

    use_adaptive_testres = False
    if args.testres == -1:
        use_adaptive_testres = True

    # construct model
    model = hsm(128, args.clean, level=args.level)
    model = nn.DataParallel(model, device_ids=[0])
    model.cuda()

    if args.loadmodel is not None:
        pretrained_dict = torch.load(args.loadmodel)
        pretrained_dict['state_dict'] = {k: v for k, v in pretrained_dict['state_dict'].items() if 'disp' not in k}
        model.load_state_dict(pretrained_dict['state_dict'], strict=False)
    else:
        print('run with random init')
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


    model.eval()

    if args.testres > 0:
        dataset = RVCDataset(args.datapath, testres=args.testres)
    else:
        dataset = RVCDataset(args.datapath, eth_testres=args.eth_testres)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    steps = 0
    for (imgL, imgR, max_disp, origianl_image_size, dataset_type , img_name) in dataloader:
        # Todo: this is a hot fix. Must be fixed to handle batchsize greater than 1
        img_name = img_name[0]


        if args.debug_image != None and not args.debug_image in img_name:
            continue

        print(img_name)

        if use_adaptive_testres:
            if dataset_type == 0: # Middlebury
                args.testres = 1
            elif dataset_type == 2:
                args.testres = 1.8
            elif dataset_type == 1: # Gengsahn said it's between 3~4. Find with linear grid search
                args.testres = 3.5
            else:
                raise ValueError("name of the folder does not contain any of: kitti, middlebury, eth3d")

        if args.max_disp > 0:
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
        print("    max disparity = " + str(model.module.maxdisp))


        ##fast pad
        max_h = int(imgL.shape[2] // 64 * 64)
        max_w = int(imgL.shape[3] // 64 * 64)
        if max_h < imgL.shape[2]: max_h += 64
        if max_w < imgL.shape[3]: max_w += 64

        wandb.log({"imgL": wandb.Image(imgL, caption=img_name + ", " + str(tuple(imgL.shape))),
                   "imgR": wandb.Image(imgR, caption=img_name + ", " + str(tuple(imgR.shape)))}, step=steps)

        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()

            pred_disp, entropy = model(imgL, imgR)

            torch.cuda.synchronize()
            ttime = (time.time() - start_time)
            torch.save(pred_disp, "/home/isaac/high-res-stereo/debug/rvc/out.pt")

            print('    time = %.2f' % (ttime * 1000))
        pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()

        top_pad = max_h - origianl_image_size[0][0]
        left_pad = max_w - origianl_image_size[1][0]
        entropy = entropy[top_pad:, :pred_disp.shape[1] - left_pad].cpu().numpy()
        pred_disp = pred_disp[top_pad:, :pred_disp.shape[1] - left_pad]

        # save predictions
        idxname = img_name
        if not os.path.exists('%s/%s' % (args.name, idxname)):
            os.makedirs('%s/%s' % (args.name, idxname))
        idxname = '%s/disp0%s' % (idxname, args.name)

        # resize to highres
        pred_disp = cv2.resize(pred_disp / args.testres, (origianl_image_size[1], origianl_image_size[0]), interpolation=cv2.INTER_LINEAR)

        # clip while keep inf
        invalid = np.logical_or(pred_disp == np.inf, pred_disp != pred_disp)
        pred_disp[invalid] = np.inf

        pred_disp_png = pred_disp / pred_disp[~invalid].max() * 255
        cv2.imwrite('%s/%s/disp.png' % (args.name, idxname.split('/')[0]),
                    pred_disp_png)
        entorpy_png = entropy / entropy.max() * 255
        cv2.imwrite('%s/%s/ent.png' % (args.name, idxname.split('/')[0]), entropy / entropy.max() * 255)

        out_pfm_path = '%s/%s.pfm' % (args.name, idxname)
        with open(out_pfm_path, 'w') as f:
            save_pfm(f, pred_disp[::-1, :])
        with open('%s/%s/time%s.txt' % (args.name, idxname.split('/')[0], args.name), 'w') as f:
            f.write(str(ttime))
        print("    output = " + out_pfm_path)

        caption = img_name + ", " + str(tuple(pred_disp_png.shape)) + ", max disparity = " +  str(max_disp) + ", time = " + str(ttime)
        wandb.log({"disparity": wandb.Image(pred_disp_png, caption=caption) , "entropy": wandb.Image(entorpy_png, caption= str(entorpy_png.shape))}, step=steps)
        torch.cuda.empty_cache()
        steps+=1



if __name__ == '__main__':
    main()

