import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath, val=False):

    left_fold  = 'colored_0/'
    right_fold = 'colored_1/'
    disp_occ   = 'disp_occ/'

    image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]

    # source: https://github.com/JiaRenChang/PSMNet/blob/master/dataloader/KITTIloader2015.py?fbclid=IwAR1n7cQaaAlVzhsvT-JvPhK6OoHasNlqaUgfQp0MfHEelX7-UOQGX9qb9a4
    train = image[:160]
    val = image[160:]

    left_val  = []
    right_val = []
    disp_val_L = []

    left_train  = [os.path.join(filepath, left_fold, img) for img in train]
    right_train = [os.path.join(filepath, right_fold, img) for img in train]
    disp_train = [os.path.join(filepath, disp_occ, img) for img in train]

    if val:
        left_val  = [os.path.join(filepath, left_fold, img) for img in val]
        right_val = [os.path.join(filepath, right_fold, img) for img in val]
        disp_val_L = [os.path.join(filepath, disp_occ, img) for img in val]

    return left_train, right_train, disp_train, left_val, right_val, disp_val_L

