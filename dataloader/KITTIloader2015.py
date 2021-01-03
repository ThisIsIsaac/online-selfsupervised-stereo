import torch.utils.data as data

import pdb
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

  left_fold  = 'image_2/'
  right_fold = 'image_3/'
  disp_L = 'disp_occ_0/'

  image = [img for img in os.listdir(os.path.join(filepath,left_fold)) if img.find('_10') > -1]
  image = sorted(image)

  # index starting from [0, 199]
  imglist = [1,3,6,20,26,35,38,41,43,44,49,60,67,70,81,84,89,97,109,119,122,123,129,130,132,134,141,144,152,158,159,165,171,174,179,182, 184,186,187,196]


  with open("dataloader/KITTI2015_val_images.txt", "w") as f:
    for idx in imglist:
      f.write(str(idx) + ": " + image[idx] + "\n")

  if not val:
    train = [image[i] for i in range(200) if i not in imglist]*100
  elif val:
    train = [image[i] for i in range(200)]*80
  val = [image[i] for i in imglist]

  left_train  = [os.path.join(filepath, left_fold, img) for img in train]
  right_train = [os.path.join(filepath, right_fold, img) for img in train]
  disp_train_L = [os.path.join(filepath, disp_L, img) for img in train]

  left_val = []
  right_val = []
  disp_val_L = []
  for img in val:
    # if img not in  ['000158_10.png', "000159_10.png", "000171_10.png", "000165_10.png", "000174_10.png", "000179_10.png"]:
    left_val.append(os.path.join(filepath, left_fold, img))
    right_val.append(os.path.join(filepath, right_fold, img))
    disp_val_L.append(os.path.join(filepath, disp_L, img))


  return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L

if __name__ == "__main__":
  dataloader("/DATA1/isaac/KITTI2015/data_scene_flow/training")