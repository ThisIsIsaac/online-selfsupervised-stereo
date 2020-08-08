import torch.utils.data as data

import pdb
from PIL import Image
import os
import os.path
import numpy as np
import glob


def mb_dataloader(filepath, val=False):
    img_list = [i.split('/')[-1] for i in glob.glob('%s/*'%filepath) if os.path.isdir(i)]
    img_list.sort()

    val_idx_list = [24, 25, 26, 27, 28, 29, 54, 55, 56, 57, 58, 59]

    left_train  = [os.path.join(filepath,img_list[idx], "im0.png") for idx in range(len(img_list)) if idx not in val_idx_list]
    right_train = [os.path.join(filepath,img_list[idx], "im1.png") for idx in range(len(img_list)) if idx not in val_idx_list]
    disp_train_L = [os.path.join(filepath,img_list[idx], "disp0GT.pfm") for idx in range(len(img_list)) if idx not in val_idx_list]
    disp_train_R = [os.path.join(filepath,img_list[idx], "disp1GT.pfm") for idx in range(len(img_list)) if idx not in val_idx_list]

    left_val  = []
    right_val = []
    disp_val_L = []
    disp_val_R = []

    if val:
        left_val  = [os.path.join(filepath,img_list[idx], "im0.png") for idx in val_idx_list]
        right_val = [os.path.join(filepath,img_list[idx], "im1.png") for idx in val_idx_list]
        disp_val_L = [os.path.join(filepath,img_list[idx], "disp0GT.pfm") for idx in val_idx_list]
        disp_val_R = [os.path.join(filepath,img_list[idx], "disp1GT.pfm") for idx in val_idx_list]

    return left_train, right_train, disp_train_L, disp_train_R, left_val, right_val, disp_val_L, disp_val_R

def hr_dataloader(filepath, val=False):
    img_list = [i.split('/')[-1] for i in glob.glob('%s/*'%filepath) if os.path.isdir(i)]
    img_list.sort()

    # 31 are handpicked, rest are randomly chosen
    val_idx_list = [16, 25, 28, 30, 31, 32, 34, 35, 43, 48, 60, 77, 84, 89, 101, 104, 105,
                    116, 132, 135, 137, 142, 153, 156, 161, 164, 165, 168, 173, 175, 183,
                    185, 191, 192, 200, 209, 211, 212, 219, 222, 231, 235, 241, 242, 244,
                    245, 246, 254, 256, 258, 260, 263, 265, 269, 273, 277, 286, 288, 290,
                    296, 304, 314, 317, 319, 321, 325, 329, 333, 336, 341, 348, 354, 361,
                    368, 369, 371, 372, 383, 394, 396, 400, 401, 423, 427, 429, 434, 437,
                    445, 446, 452, 457, 463, 472, 481, 485, 486, 490, 496, 497, 507, 517,
                    528, 529, 533, 539, 540, 542, 548, 551, 563, 567, 570, 572, 575, 577,
                    578, 579, 609, 611, 616, 617, 624, 628, 630, 636, 637, 650, 654, 655,
                    664, 666, 673, 684, 685, 687, 689, 692, 700, 701, 704, 705, 711, 719,
                    720, 721, 731, 734, 739, 740, 743, 749, 753, 755, 765, 767, 778]

    left_val  = []
    right_val = []
    disp_val_L = []
    disp_val_R = []

    if val:
        left_val  = [os.path.join(filepath,img_list[idx], "im0.png") for idx in val_idx_list]
        right_val = [os.path.join(filepath,img_list[idx], "im1.png") for idx in val_idx_list]
        disp_val_L = [os.path.join(filepath,img_list[idx], "disp0GT.pfm") for idx in val_idx_list]
        disp_val_R = [os.path.join(filepath,img_list[idx], "disp1GT.pfm") for idx in val_idx_list]

    left_train  = [os.path.join(filepath,img_list[idx], "im0.png") for idx in range(len(img_list)) if idx not in val_idx_list]
    right_train = [os.path.join(filepath,img_list[idx], "im1.png") for idx in range(len(img_list)) if idx not in val_idx_list]
    disp_train_L = [os.path.join(filepath,img_list[idx], "disp0GT.pfm") for idx in range(len(img_list)) if idx not in val_idx_list]
    disp_train_R = [os.path.join(filepath,img_list[idx], "disp1GT.pfm") for idx in range(len(img_list)) if idx not in val_idx_list]

    return left_train, right_train, disp_train_L, disp_train_R, left_val, right_val, disp_val_L, disp_val_R


def eth_dataloader(filepath, val=False):
    img_path = os.path.join(filepath, "two_view_training")
    img_list = [i.split('/')[-1] for i in glob.glob('%s/*'%img_path) if os.path.isdir(i)]
    img_list.sort()

    # handpicked
    val_idx = [0, 1, 8, 9, 13, 17, 18]
    left_val  = []
    right_val = []
    disp_val_L = []

    disp_path = os.path.join(filepath, "two_view_training_gt")
    disp_list = [i.split('/')[-1] for i in glob.glob('%s/*'%disp_path) if os.path.isdir(i)]
    disp_list.sort()

    if val:
        left_val = [os.path.join(img_path,img_list[idx], "im0.png") for idx in val_idx]
        left_val = [os.path.join(img_path,img_list[idx], "im1.png") for idx in val_idx]
        disp_val_L = [os.path.join(disp_path,disp_list[idx], "disp0GT.pfm") for idx in val_idx]

    left_train  = [os.path.join(img_path,img_list[idx], "im0.png") for idx in range(len(img_list)) if idx not in val_idx]
    right_train = [os.path.join(img_path,img_list[idx], "im1.png") for idx in range(len(img_list)) if idx not in val_idx]
    disp_train_L = [os.path.join(disp_path,disp_list[idx], "disp0GT.pfm") for idx in range(len(img_list)) if idx not in val_idx]

    return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L


