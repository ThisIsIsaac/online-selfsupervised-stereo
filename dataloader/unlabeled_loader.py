from numpy.lib.function_base import disp
from torch.utils.data import Dataset
import skimage.io
import os
from utils.preprocess import get_transform
import cv2
import torch
import numpy as np
from utils.readpfm import readPFM

class RVCDataset(Dataset):
def __init__(self, args):

    disp_dir = "/home/isaac/high-res-stereo/unlabeled_util/pseudo_gt/disp/"
    entropy_dir = "/home/isaac/high-res-stereo/unlabeled_util/pseudo_gt/entropy/"

    disp_paths = []
    entropy_paths = []

    for img in os.listdir(disp_dir):
        disp_paths.append(os.path.join(disp_dir, img))
        entropy_paths.append(os.path.join(entropy_dir, img))

return disp_paths, entropy_paths

    def __len__(self):
        return len(self.disp_paths)

    def __getitem__(self, item):
        index = item
        for (key, val) in self.stereo_pair_subfolders.items():
            if index >= len(val):
                index = index - len(val)

                if index < 0:
                    raise ValueError("Something went wrong during index calculation")

            else:
                break
        folder = key
        img_name = val[index]
        dataset_type=-1
        if self.variable_testres and self.testres == -1:
            if "middlebury" in img_name.lower():  # Middlebury
                dataset_type = 0
                testres = self.middlebury_testres
            elif "kitti" in img_name.lower():  # Kitti
                dataset_type = 2
                testres = self.kitti_testres
            elif "eth3d" in img_name.lower():  # ETH: Gengsahn said it's between 3~4. Find with linear grid search
                dataset_type = 1
                testres = self.eth_testres
            else:
                raise ValueError("name of the folder does not contain any of: kitti, middlebury, eth3d")
        else:
            testres = self.testres

        im0_path = os.path.join(folder, img_name, 'im0.png')
        im1_path = os.path.join(folder, img_name, 'im1.png')
        imgL_o = (skimage.io.imread(im0_path).astype('float32'))[:, :, :3] # (375, 1242, 3)
        imgR_o = (skimage.io.imread(im1_path).astype('float32'))[:, :, :3]
        original_image_size = imgL_o.shape[:2]

        gt_path = os.path.join(folder, img_name, "disp0GT.pfm")
        gt_disp_raw = readPFM(gt_path)[0].copy() # hotfix to prevent error when converting np.array with negative stride to Tensor
        # [375 x 1242]
        # np.inf == 353307
        # np.NINF == 0

        path_to_replace = os.path.basename(os.path.normpath(im0_path))
        with open(im0_path.replace(path_to_replace, 'calib.txt')) as f:
            lines = f.readlines()
            max_disp = int(int(lines[6].split('=')[-1]))



        imgL_o = cv2.resize(imgL_o, None, fx=testres, fy=testres, interpolation=cv2.INTER_CUBIC) # [675 x 2236]
        imgR_o = cv2.resize(imgR_o, None, fx=testres, fy=testres, interpolation=cv2.INTER_CUBIC)
        #gt_disp_raw = cv2.resize(gt_disp_raw, None, fx=testres, fy=testres, interpolation=cv2.INTER_CUBIC) # [675 x 2236]
        # print(dict(zip(*np.unique(gt_disp_raw, return_counts=True)))[np.inf])
        # print(dict(zip(*np.unique(gt_disp_raw, return_counts=True)))[np.NINF])
        # [675, 2236] min=4.5859375, max=inf

        # if self.debug:
        #     input_img_path = '%s/%s/' % (self.args.name, img_name)
        #     assert (cv2.imwrite(input_img_path + "imgR.png", imgR_o))
        #     assert (cv2.imwrite(input_img_path + "imgL.png", imgL_o))


        imgL = self.transform(imgL_o).numpy()
        imgR = self.transform(imgR_o).numpy()

        imgL = np.reshape(imgL, [3, imgL.shape[1], imgL.shape[2]])
        imgR = np.reshape(imgR, [3, imgR.shape[1], imgR.shape[2]])

        ##fast pad
        max_h = int(imgL.shape[1] // 64 * 64)
        max_w = int(imgL.shape[2] // 64 * 64)
        if max_h < imgL.shape[1]: max_h += 64
        if max_w < imgL.shape[2]: max_w += 64

        top_pad = max_h - imgL.shape[1]
        left_pad = max_w - imgL.shape[2]
        imgL = np.lib.pad(imgL, ( (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)
        imgR = np.lib.pad(imgR, ( (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)

        # test
        imgL = torch.FloatTensor(imgL)
        imgR = torch.FloatTensor(imgR)

        return (imgL, imgR, gt_disp_raw, max_disp, original_image_size, top_pad, left_pad, testres, dataset_type , os.path.join(folder, img_name))