import argparse
import cv2
from models import hsm
import os
import torch.backends.cudnn as cudnn
import time
from models.submodule import *
from utils.eval import save_pfm
from dataloader.RVCDataset import RVCDataset
from torch.utils.data import DataLoader
cudnn.benchmark = False
import skimage
import wandb
import score_rvc
from wandb import magic
from utils.readpfm import readPFM
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from colorspacious import cspace_converter
from utils.disp_converter import convert_to_colormap

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
    parser.add_argument("--eth_testres" , type=float, default=3.5)
    parser.add_argument("--score_results", action="store_true", default=False)
    parser.add_argument("--save_weights", action="store_true", default=False)
    parser.add_argument("--kitti", action="store_true", default=False)
    parser.add_argument("--eth", action="store_true", default=False)
    parser.add_argument("--mb", action="store_true", default=False)
    parser.add_argument("--all_data", action="store_true", default=False)
    parser.add_argument("--eval_train_only", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    wandb.init(name=args.name, project="rvc_stereo", save_code=True, magic=True, config=args, dir="/tmp")

    if not os.path.exists("output"):
        os.mkdir("output")

    kitti_merics = {}
    eth_metrics = {}
    mb_metrics = {}

    # construct model
    model = hsm(128, args.clean, level=args.level)
    wandb.watch(model)
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

    dataset = RVCDataset(args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    steps = 0
    for (imgL, imgR, gt_disp_raw, max_disp, origianl_image_size, top_pad, left_pad, testres, dataset_type , data_path) in dataloader:
        # Todo: this is a hot fix. Must be fixed to handle batchsize greater than 1
        data_path = data_path[0]
        img_name = os.path.basename(os.path.normpath(data_path))
        testres = float(testres[0])
        gt_disp_raw = gt_disp_raw[0]

        cum_metrics = None
        if dataset_type == 0:
            cum_metrics = mb_metrics

        elif dataset_type == 1:
            cum_metrics = eth_metrics

        elif dataset_type == 2:
            cum_metrics = kitti_merics

        print(img_name)

        if args.max_disp > 0:
            max_disp = int(args.max_disp)

        ## change max disp
        tmpdisp = int(max_disp * testres // 64 * 64)
        if (max_disp * testres / 64 * 64) > tmpdisp:
            model.module.maxdisp = tmpdisp + 64
        else:
            model.module.maxdisp = tmpdisp
        if model.module.maxdisp == 64: model.module.maxdisp = 128
        model.module.disp_reg8 = disparityregression(model.module.maxdisp, 16).cuda()
        model.module.disp_reg16 = disparityregression(model.module.maxdisp, 16).cuda()
        model.module.disp_reg32 = disparityregression(model.module.maxdisp, 32).cuda()
        model.module.disp_reg64 = disparityregression(model.module.maxdisp, 64).cuda()
        print("    max disparity = " + str(model.module.maxdisp))


        # ##fast pad
        # max_h = int(imgL.shape[2] // 64 * 64)
        # max_w = int(imgL.shape[3] // 64 * 64)
        # if max_h < imgL.shape[2]: max_h += 64
        # if max_w < imgL.shape[3]: max_w += 64

        wandb.log({"imgL": wandb.Image(imgL, caption=img_name + ", " + str(tuple(imgL.shape))),
                   "imgR": wandb.Image(imgR, caption=img_name + ", " + str(tuple(imgR.shape)))}, step=steps)

        # if args.debug:
        #     input_img_path = '%s/%s/' % (args.name, img_name)
        #     assert(cv2.imwrite(input_img_path+"imgR.png", imgR[0].permute(1, 2, 0).numpy()))
        #     assert(cv2.imwrite(input_img_path + "imgL.png", imgL[0].permute(1, 2, 0).numpy()))

        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()

            pred_disp, entropy = model(imgL, imgR)

            torch.cuda.synchronize()
            ttime = (time.time() - start_time)

            print('    time = %.2f' % (ttime * 1000))
        pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()

        # top_pad = max_h - origianl_image_size[0][0]
        # left_pad = max_w - origianl_image_size[1][0]
        top_pad = int(top_pad[0])
        left_pad = int(left_pad[0])
        entropy = entropy[top_pad:, :pred_disp.shape[1] - left_pad].cpu().numpy()
        pred_disp = pred_disp[top_pad:, :pred_disp.shape[1] - left_pad]

        # save predictions
        idxname = img_name

        if not os.path.exists('output/%s/%s' % (args.name, idxname)):
            os.makedirs('output/%s/%s' % (args.name, idxname))

        idxname = '%s/disp0%s' % (idxname, args.name)

        # resize to highres
        pred_disp_raw = cv2.resize(pred_disp / testres, (origianl_image_size[1], origianl_image_size[0]), interpolation=cv2.INTER_LINEAR)
        pred_disp = pred_disp_raw # raw is to use for scoring

        gt_disp = gt_disp_raw.numpy()

        # clip while keep inf
        pred_invalid = np.logical_or(pred_disp == np.inf, pred_disp != pred_disp)
        pred_disp[pred_invalid] = np.inf
        pred_disp_png = (pred_disp).astype("uint16")

        gt_invalid = np.logical_or(gt_disp == np.inf, gt_disp != gt_disp)
        gt_disp[gt_invalid] = np.inf
        gt_disp_png = (gt_disp).astype("uint16")

        entorpy_png = (entropy).astype('uint16')

        # to increase contrast in the png images
        # contrast = 10
        # gt_disp_png *= contrast
        # pred_disp_png *= contrast
        # entorpy_png *= contrast

        pred_disp_path = 'output/%s/%s/disp.png' % (args.name, idxname.split('/')[0])
        gt_disp_path = 'output/%s/%s/gt_disp.png' % (args.name, idxname.split('/')[0])
        

        # ! Experimental color maps
        gt_disp_color_path = 'output/%s/%s/gt_disp_color.png' % (args.name, idxname.split('/')[0])
        pred_disp_color_path = 'output/%s/%s/disp_color.png' % (args.name, idxname.split('/')[0])
        diff_disp_color_path = 'output/%s/%s/diff_color.png' % (args.name, idxname.split('/')[0])
        gt_colormap = convert_to_colormap(gt_disp_png)
        pred_colormap = convert_to_colormap(pred_disp_png)
        diff_colormap = abs(gt_colormap - pred_colormap)
        # plt.get_cmap("plasma")

        assert(cv2.imwrite(gt_disp_color_path, gt_colormap))
        assert(cv2.imwrite(pred_disp_color_path, pred_colormap))
        assert(cv2.imwrite(diff_disp_color_path, diff_colormap))
        # docs: https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=imwrite#imwrite
        
        assert(cv2.imwrite(pred_disp_path, pred_disp_png))
        assert(cv2.imwrite(gt_disp_path, gt_disp_png))
        # get rid of dividing by max
        assert(cv2.imwrite('output/%s/%s/ent.png' % (args.name, idxname.split('/')[0]), entorpy_png))

        out_pfm_path = 'output/%s/%s.pfm' % (args.name, idxname)
        with open(out_pfm_path, 'w') as f:
            save_pfm(f, pred_disp[::-1, :])
        with open('output/%s/%s/time_%s.txt' % (args.name, idxname.split('/')[0], args.name), 'w') as f:
            f.write(str(ttime))
        print("    output = " + out_pfm_path)

        caption = img_name + ", " + str(tuple(pred_disp_png.shape)) + ", max disparity = " +  str(int(max_disp[0])) + ", time = " + str(ttime)

        # read GT depthmap and upload as jpg
        wandb.log({"pred": wandb.Image(pred_disp_png, caption=caption) , "gt": wandb.Image(gt_disp_png), "diff_color":wandb.Image(diff_colormap), 
        "entropy": wandb.Image(entorpy_png, caption= str(entorpy_png.shape)),  "pred_color": wandb.Image(pred_colormap, caption=caption) , "gt_color": wandb.Image(gt_colormap)}, step=steps)
        torch.cuda.empty_cache()
        steps+=1

        # Todo: find out what mask0nocc does. It's probably not the same as KITTI's object map
        if dataset_type == 2:
            obj_map_path = os.path.join(data_path, "obj_map.png")
        else:
            obj_map_path = None

        if args.score_results:
            if pred_disp_raw.shape != gt_disp_raw.shape: # pred_disp_raw[375 x 1242] gt_disp_raw[675 x 2236]
                ratio = float(gt_disp_raw.shape[1]) / pred_disp_raw.shape[1]
                disp_resized = cv2.resize(pred_disp_raw, (gt_disp_raw.shape[1], gt_disp_raw.shape[0])) * ratio
                pred_disp_raw = disp_resized # [675 x 2236]
            if args.debug:
                out_resized_pfm_path = 'output/%s/%s/pred_scored.pfm' % (args.name, img_name)
                with open(out_resized_pfm_path, 'w') as f:
                    save_pfm(f, pred_disp_raw)

                out_resized_gt_path = 'output/%s/%s/gt_scored.pfm' % (args.name, img_name)
                with open(out_resized_gt_path, 'w') as f:
                    save_pfm(f, gt_disp_raw.numpy())
                    # [675, 2236] np.inf == 1464079, np.NINF == 4875

            # (disp, gt, max_disp, datatype, save_path, disp_path=None, gt_path=None, obj_map_path=None, ABS_THRESH=3.0, REL_THRESH=0.05)
            metrics = score_rvc.get_metrics(pred_disp_raw, gt_disp_raw, int(max_disp[0]), dataset_type, ('output/%s/%s' % (args.name, idxname.split('/')[0])), disp_path=pred_disp_path, gt_path=gt_disp_path, obj_map_path=obj_map_path, debug=args.debug)

            avg_metrics = {}
            for (key, val) in metrics.items():
                if cum_metrics.get(key) == None:
                    cum_metrics[key] = []
                cum_metrics[key].append(val)
                avg_metrics["avg_" + key] = sum(cum_metrics[key]) / len(cum_metrics[key])

            wandb.log(metrics, step=steps)
            wandb.log(avg_metrics, step=steps)

    if args.save_weights and os.path.exists(args.loadmodel):
        wandb.save(args.loadmodel)

if __name__ == '__main__':
    main()

