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
# import wandb
import score_rvc
from utils.disp_converter import convert_to_colormap
from sync_batchnorm.sync_batchnorm import convert_model
from utils.prepare_submission import prepare_kitti
import subprocess
from dataloader import KITTIloader2015 as lk15
from dataloader import MiddleburyLoader as DA
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
    parser.add_argument('--testres', type=float, default=0.5, #default used to be 0.5
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
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--prepare_kitti", action="store_true", default=False)

    args = parser.parse_args()

    # wandb.init(name=args.name, project="high-res-stereo", save_code=True, magic=True, config=args)

    if not os.path.exists("output"):
        os.mkdir("output")

    kitti_merics = {}
    eth_metrics = {}
    mb_metrics = {}

    # construct model
    model = hsm(128, args.clean, level=args.level)
    model = convert_model(model)
    # wandb.watch(model)
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

    if not args.prepare_kitti:
        dataset = RVCDataset(args)
    if args.prepare_kitti:
        _, _, _, left_val, right_val, disp_val_L = lk15.dataloader('/data/private/KITTI2015/data_scene_flow/training/',
                                                                      val=True)  # change to trainval when finetuning on KITTI

        dataset = DA.myImageFloder(left_val, right_val, disp_val_L, rand_scale=[1,1], order=0)
        
    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)
        
    steps = 0
    max_disp = None
    origianl_image_size= None
    top_pad=None
    left_pad=None
    testres=[args.testres]
    dataset_type=None
    data_path=[args.datapath]
    # for (imgL, imgR, gt_disp_raw, max_disp, origianl_image_size, top_pad, left_pad, testres, dataset_type , data_path) in dataloader:
    for (imgL, imgR, gt_disp_raw) in dataloader:
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

        # wandb.log({"imgL": wandb.Image(imgL, caption=img_name + ", " + str(tuple(imgL.shape))),
        #            "imgR": wandb.Image(imgR, caption=img_name + ", " + str(tuple(imgR.shape)))}, step=steps)

        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()

            # * output dimensions same as input dimensions
            # * (ex: imgL[1, 3, 704, 2240] then pred_disp[1, 704, 2240])
            pred_disp, entropy = model(imgL, imgR)

            torch.cuda.synchronize()
            ttime = (time.time() - start_time)

            print('    time = %.2f' % (ttime * 1000))

        # * squeeze (remove dimensions with size 1) (ex: pred_disp[1, 704, 2240] ->[704, 2240])
        pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()

        top_pad = int(top_pad[0])
        left_pad = int(left_pad[0])
        entropy = entropy[top_pad:, :pred_disp.shape[1] - left_pad].cpu().numpy()
        pred_disp = pred_disp[top_pad:, :pred_disp.shape[1] - left_pad]

        # save predictions
        idxname = img_name

        if not os.path.exists('output/%s/%s' % (args.name, idxname)):
            os.makedirs('output/%s/%s' % (args.name, idxname))

        idxname = '%s/disp0%s' % (idxname, args.name)

        # * shrink image back to the GT size (ex: pred_disp[675, 2236] -> [375, 1242])
        # ! we element-wise divide pred_disp by testres becasue the image is shrinking,
        # ! so the distance between pixels should also shrink by the same factor
        pred_disp_raw = cv2.resize(pred_disp / testres, (origianl_image_size[1], origianl_image_size[0]), interpolation=cv2.INTER_LINEAR)
        pred_disp = pred_disp_raw # raw is to use for scoring

        gt_disp = gt_disp_raw.numpy()

        # * clip while keep inf
        # ? `pred_disp != pred_disp` is always true, right??
        # ? `pred_disp[pred_invalid] = np.inf` why do this?
        pred_invalid = np.logical_or(pred_disp == np.inf, pred_disp != pred_disp)
        pred_disp[pred_invalid] = np.inf

        pred_disp_png = (pred_disp*256).astype("uint16")

        gt_invalid = np.logical_or(gt_disp == np.inf, gt_disp != gt_disp)
        gt_disp[gt_invalid] = 0
        gt_disp_png = (gt_disp*256).astype("uint16")
        entorpy_png = (entropy*256).astype('uint16')

        # ! raw output to png
        pred_disp_path = 'output/%s/%s/disp.png' % (args.name, idxname.split('/')[0])
        gt_disp_path = 'output/%s/%s/gt_disp.png' % (args.name, idxname.split('/')[0])
        assert(cv2.imwrite(pred_disp_path, pred_disp_png))
        assert(cv2.imwrite(gt_disp_path, gt_disp_png))
        assert(cv2.imwrite('output/%s/%s/ent.png' % (args.name, idxname.split('/')[0]), entorpy_png))

        # ! Experimental color maps
        gt_disp_color_path = 'output/%s/%s/gt_disp_color.png' % (args.name, idxname.split('/')[0])
        pred_disp_color_path = 'output/%s/%s/disp_color.png' % (args.name, idxname.split('/')[0])

        gt_colormap = convert_to_colormap(gt_disp_png)
        pred_colormap = convert_to_colormap(pred_disp_png)
        entropy_colormap = convert_to_colormap(entorpy_png)
        assert(cv2.imwrite(gt_disp_color_path, gt_colormap))
        assert(cv2.imwrite(pred_disp_color_path, pred_colormap))


        # ! diff colormaps
        diff_colormap_path = 'output/%s/%s/diff_color.png' % (args.name, idxname.split('/')[0])
        false_positive_path = 'output/%s/%s/false_positive_color.png' % (args.name, idxname.split('/')[0])
        false_negative_path = 'output/%s/%s/false_negative_color.png' % (args.name, idxname.split('/')[0])
        gt_disp_png[gt_invalid] = pred_disp_png[gt_invalid]
        gt_disp_png = gt_disp_png.astype("int32")
        pred_disp_png = pred_disp_png.astype("int32")

        diff_colormap = convert_to_colormap(np.abs(gt_disp_png - pred_disp_png))
        false_positive_colormap = convert_to_colormap(np.abs(np.clip(gt_disp_png - pred_disp_png, None, 0)))
        false_negative_colormap = convert_to_colormap(np.abs(np.clip(gt_disp_png - pred_disp_png, 0, None)))
        assert(cv2.imwrite(diff_colormap_path, diff_colormap))
        assert(cv2.imwrite(false_positive_path, false_positive_colormap))
        assert(cv2.imwrite(false_negative_path, false_negative_colormap))



        out_pfm_path = 'output/%s/%s.pfm' % (args.name, idxname)
        with open(out_pfm_path, 'w') as f:
            save_pfm(f, pred_disp[::-1, :])
        with open('output/%s/%s/time_%s.txt' % (args.name, idxname.split('/')[0], args.name), 'w') as f:
            f.write(str(ttime))
        print("    output = " + out_pfm_path)

        caption = img_name + ", " + str(tuple(pred_disp_png.shape)) + ", max disparity = " +  str(int(max_disp[0])) + ", time = " + str(ttime)

        # read GT depthmap and upload as jpg

        # wandb.log({"disparity": wandb.Image(pred_colormap, caption=caption) , "gt": wandb.Image(gt_colormap), "entropy": wandb.Image(entropy_colormap, caption= str(entorpy_png.shape)),
        #            "diff":wandb.Image(diff_colormap), "false_positive":wandb.Image(false_positive_colormap), "false_negative":wandb.Image(false_negative_colormap)}, step=steps)

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
            # if args.debug:
            #     out_resized_pfm_path = 'output/%s/%s/pred_scored.pfm' % (args.name, img_name)
            #     with open(out_resized_pfm_path, 'w') as f:
            #         save_pfm(f, pred_disp_raw)

            #     out_resized_gt_path = 'output/%s/%s/gt_scored.pfm' % (args.name, img_name)
            #     with open(out_resized_gt_path, 'w') as f:
            #         save_pfm(f, gt_disp_raw.numpy())

            metrics = score_rvc.get_metrics(pred_disp_raw, gt_disp_raw, int(max_disp[0]), dataset_type, ('output/%s/%s' % (args.name, idxname.split('/')[0])), disp_path=pred_disp_path, gt_path=gt_disp_path, obj_map_path=obj_map_path, debug=args.debug)

            avg_metrics = {}
            for (key, val) in metrics.items():
                if cum_metrics.get(key) == None:
                    cum_metrics[key] = []
                cum_metrics[key].append(val)
                avg_metrics["avg_" + key] = sum(cum_metrics[key]) / len(cum_metrics[key])

            # wandb.log(metrics, step=steps)
            # wandb.log(avg_metrics, step=steps)

    # if args.save_weights and os.path.exists(args.loadmodel):
    #     wandb.save(args.loadmodel)

    if args.prepare_kitti and (args.all_data or args.kitti):
        in_path = 'output/%s' % (args.name)
        out_path = "/home/isaac/high-res-stereo/kitti_submission_output"
        out_path = prepare_kitti(in_path,  out_path)
        subprocess.run(["/home/isaac/KITTI2015_devkit/cpp/eval_scene_flow", out_path])
        print("KITTI submission evaluation saved to: " + out_path)
if __name__ == '__main__':
    main()

