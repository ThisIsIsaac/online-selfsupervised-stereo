from torch.utils.data import Dataset
import skimage
import os
from utils.preprocess import get_transform
import cv2
import torch
import numpy as np
import wandb

class RVCDataset(Dataset):
    def __init__(self, data_path, eth_testres=3.5, wandb_logger=None, include_training=True, include_test=True, testres=None):
        self.folders = []
        self.stereo_pair_subfolders = []
        self.kitti_testres = 1.8
        self.middlebury_testres = 1
        self.eth_testres = eth_testres
        self.transform = get_transform()
        self.wandb_logger = wandb_logger
        self.variable_testres=(testres==None)
        self.testres = testres

        if include_training:
            self.folders.append(os.path.join(data_path, 'training'))
        if include_test:
            self.folders.append(os.path.join(data_path, 'test'))

        self.stereo_pair_subfolders = {}
        self.total_len = 0
        for folder in self.folders:
            self.stereo_pair_subfolders[folder] = []
            datasets = [dataset for dataset
                        in os.listdir(folder)
                        if os.path.isdir(os.path.join(folder, dataset))]

            for dataset_name in datasets:
                self.stereo_pair_subfolders[folder].append(dataset_name)
            self.total_len += len(self.stereo_pair_subfolders[folder])

    def __len__(self):
        return self.total_len

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
        if self.variable_testres and self.testres == None:
            if "middlebury" in img_name.lower():  # Middlebury
                dataset_type = 0
                testres = self.middlebury_testres
            elif "kitti" in img_name.lower():  # ETH
                dataset_type = 1
                testres = self.kitti_testres
            elif "eth3d" in img_name.lower():  # Gengsahn said it's between 3~4. Find with linear grid search
                dataset_type = 2
                testres = self.eth_testres
            else:
                raise ValueError("name of the folder does not contain any of: kitti, middlebury, eth3d")
        else:
            testres = self.testres

        im0_path = os.path.join(folder, img_name, 'im0.png')
        im1_path = os.path.join(folder, img_name, 'im1.png')
        imgL_o = (skimage.io.imread(im0_path).astype('float32'))[:, :, :3]
        imgR_o = (skimage.io.imread(im1_path).astype('float32'))[:, :, :3]

        # wandb.log({"raw_imgL": wandb.Image(imgL_o, caption=str(imgL_o.shape)),"raw_imgR":wandb.Image(imgR_o, caption=str(imgR_o.shape))}, step=item)

        original_image_size = imgL_o.shape[:2]

        path_to_replace = os.path.basename(os.path.normpath(im0_path))
        with open(im0_path.replace(path_to_replace, 'calib.txt')) as f:
            lines = f.readlines()
            max_disp = int(int(lines[6].split('=')[-1]))

        imgL_o = cv2.resize(imgL_o, None, fx=testres, fy=testres, interpolation=cv2.INTER_CUBIC)
        imgR_o = cv2.resize(imgR_o, None, fx=testres, fy=testres, interpolation=cv2.INTER_CUBIC)
        # imgL = self.transform(imgL_o).numpy()
        # imgR = self.transform(imgR_o).numpy()
        imgL = imgL_o
        imgR = imgR_o

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

        # wandb.log({"imgL": wandb.Image(imgL, caption=str(imgL.shape)), "imgR": wandb.Image(imgR, caption=str(imgR.shape))}, step=item)

        return (imgL, imgR, max_disp, original_image_size, dataset_type , img_name)