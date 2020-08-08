import torch.utils.data as data

import pdb
from PIL import Image
import os
import os.path
import numpy as np
import glob


def dataloader(filepath):
  img_list = [i.split('/')[-1] for i in glob.glob('%s/*'%filepath) if os.path.isdir(i)]

  left_train  = ['%s/%s/im0.png'% (filepath,img) for img in img_list]  
  right_train = ['%s/%s/im1.png'% (filepath,img) for img in img_list]
  disp_train_L = ['%s/%s/disp0GT.pfm' % (filepath,img) for img in img_list]
  disp_train_R = ['%s/%s/disp1GT.pfm' % (filepath,img) for img in img_list]

  return left_train, right_train, disp_train_L, disp_train_R

def eth_dataloader(filepath):
  img_path = os.path.join(filepath, "two_view_training")
  img_list = [i.split('/')[-1] for i in glob.glob('%s/*'%img_path) if os.path.isdir(i)]

  disp_path = os.path.join(filepath, "two_view_training_gt")
  disp_list = [i.split('/')[-1] for i in glob.glob('%s/*'%disp_path) if os.path.isdir(i)]

  left_train  = ['%s/%s/im0.png'% (filepath,img) for img in img_list]
  right_train = ['%s/%s/im1.png'% (filepath,img) for img in img_list]
  disp_train_L = ['%s/%s/disp0GT.pfm' % (filepath,img) for img in disp_list]
  disp_train_R = ['%s/%s/disp1GT.pfm' % (filepath,img) for img in disp_list]

  return left_train, right_train, disp_train_L, disp_train_R
