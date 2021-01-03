import os, torch, torch.utils.data as data
from PIL import Image
import numpy as np
from utils import preprocess
from utils import readpfm as rp
from . import flow_transforms
import pdb
import torchvision
import warnings
from utils.preprocess import get_transform
warnings.filterwarnings('ignore', '.*output shape of zoom.*')
import cv2 

IMG_EXTENSIONS = [
 '.jpg', '.JPG', '.jpeg', '.JPEG',
 '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_image_file(filename):
    return any((filename.endswith(extension) for extension in IMG_EXTENSIONS))


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):

    if path.endswith('.png'):
        data = Image.open(path)
        data = np.ascontiguousarray(data,dtype=np.float32)/256
        out = data
    elif path.endswith(".pfm"):
        out = rp.readPFM(path)[0]
    elif path.endswith(".npy"):
        out = np.load(path)
    else:
        raise ValueError("Disparity file is in unrecognized format: " + path)

    for s in out.strides:
        if s < 0:
            out = out.copy()
        break
    return out

class myImageFloder(data.Dataset):

    def __init__(self, left, right, left_disparity, right_disparity=None, left_entropy=None, is_validation=False,
                 loader=default_loader, dploader=disparity_loader, rand_scale=[0.225,0.6], rand_bright=[0.5,2.],
                 order=0, entropy_threshold=None, testres=None, use_pseudoGT=False, no_aug=False):
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.disp_R = right_disparity
        self.loader = loader
        self.dploader = dploader
        self.rand_scale = rand_scale
        self.rand_bright = rand_bright
        self.order = order
        self.left_entropy = left_entropy
        self.entropy_threshold = entropy_threshold
        self.is_validation = is_validation
        self.processed = get_transform()
        self.testres = testres
        self.no_aug = no_aug
        self.use_pseudoGT = use_pseudoGT

        if self.is_validation and self.testres is None:
            raise ValueError("testres argument is required for validation")

        if self.use_pseudoGT:
            if entropy_threshold is None or left_entropy is None:
                raise ValueError("when using pseudo GT, entorpy_threshold and left_entropy must be provided")


    def __getitem__(self, index):

        left = self.left[index]
        right = self.right[index]
        left_img = self.loader(left)
        right_img = self.loader(right)
        disp_L = self.disp_L[index]
        dataL = self.dploader(disp_L)
        # I accidently turned 2D disp to 3D. But the channels have the same info, so for a quick patch, I will just use one of the three
        if len(dataL.shape) == 3:
            dataL = dataL[:, :, 0]
        dataL[dataL == np.inf] = 0

        ##* training *##
        if not self.is_validation:
            if self.use_pseudoGT:
                entropy = self.dploader(self.left_entropy[index])
                entropy = cv2.resize(entropy, (dataL.shape[1], dataL.shape[0]), interpolation=cv2.INTER_LINEAR)
                mask = entropy > self.entropy_threshold
                dataL[mask] = 0

            if self.disp_R is not None:
                disp_R = self.disp_R[index]
                dataR = self.dploader(disp_R)
                dataR[dataR == np.inf] = 0

            max_h = 2048 // 4
            max_w = 3072 // 4





            #* don't perform any augmentation
            if self.no_aug:
                # print("no data augmentation selected")
                h, w = dataL.shape
                new_H = int(max_h)
                new_W = int(w * (new_H / h))
                left_img = torch.from_numpy(np.asarray(left_img)).permute(2, 0, 1)
                right_img = torch.from_numpy(np.asarray(right_img)).permute(2, 0, 1)
                dataL = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(np.asarray(dataL)), dim=0), dim=0)

                dataL =torch.nn.functional.interpolate(dataL, (new_H,new_W))
                augmented = torch.nn.functional.interpolate(torch.stack([left_img, right_img], axis=0), (new_H,new_W))
                # augmented = torchvision.transforms.functional.resize([left_img, right_img], (max_h,max_w))
                f = torchvision.transforms.CenterCrop((max_h, max_w))
                dataL = f(dataL)
                augmented = f(augmented)
                # right_img = np.asarray(right_img)
                # left_img = torchvision.transforms.functional.adjust_brightness(left_img, 0.95)
                # left_img = np.asarray(left_img)
                # angle = 0;
                # px = 0
                # if np.random.binomial(1,0.5):
                #     angle=0.1;px=2
                # co_transform = flow_transforms.Compose([
                #     flow_transforms.RandomCrop((max_h,max_w))
                #     ])

                augmented += torch.rand(augmented.shape)> 0.5

                # augmented,dataL = co_transform([left_img, right_img], dataL)
                left_img = augmented[0]
                right_img = augmented[1]

            if not self.no_aug:
                # photometric unsymmetric-augmentation
                random_brightness = np.random.uniform(self.rand_bright[0], self.rand_bright[1],2)
                random_gamma = np.random.uniform(0.8, 1.2,2)
                random_contrast = np.random.uniform(0.8, 1.2,2)
                left_img = torchvision.transforms.functional.adjust_brightness(left_img, random_brightness[0])
                left_img = torchvision.transforms.functional.adjust_gamma(left_img, random_gamma[0])
                left_img = torchvision.transforms.functional.adjust_contrast(left_img, random_contrast[0])
                right_img = torchvision.transforms.functional.adjust_brightness(right_img, random_brightness[1])
                right_img = torchvision.transforms.functional.adjust_gamma(right_img, random_gamma[1])
                right_img = torchvision.transforms.functional.adjust_contrast(right_img, random_contrast[1])
                right_img = np.asarray(right_img)
                left_img = np.asarray(left_img)

                # horizontal flip
                if not (self.disp_R is None):
                    if np.random.binomial(1,0.5):
                        tmp = right_img
                        right_img = left_img[:,::-1]
                        left_img = tmp[:,::-1]
                        tmp = dataR
                        dataR = dataL[:,::-1]
                        dataL = tmp[:,::-1]

                # geometric unsymmetric-augmentation
                angle=0;px=0
                if np.random.binomial(1,0.5):
                    angle=0.1;px=2
                co_transform = flow_transforms.Compose([
                    flow_transforms.RandomVdisp(angle,px),
                    flow_transforms.Scale(np.random.uniform(self.rand_scale[0],self.rand_scale[1]),order=self.order),
                    flow_transforms.RandomCrop((max_h,max_w)),
                    ])
                augmented,dataL = co_transform([left_img, right_img], dataL)
                left_img = augmented[0]
                right_img = augmented[1]


                # randomly occlude a region
                if np.random.binomial(1,0.5):
                    sx = int(np.random.uniform(50,150))
                    sy = int(np.random.uniform(50,150))
                    cx = int(np.random.uniform(sx,right_img.shape[0]-sx))
                    cy = int(np.random.uniform(sy,right_img.shape[1]-sy))
                    right_img[cx-sx:cx+sx,cy-sy:cy+sy] = np.mean(np.mean(right_img,0),0)[np.newaxis,np.newaxis]

                h, w,_ = left_img.shape
                top_pad = max_h - h
                left_pad = max_w - w
                left_img = np.lib.pad(left_img, ((top_pad, 0), (0, left_pad),(0,0)), mode='constant', constant_values=0)
                right_img = np.lib.pad(right_img, ((top_pad, 0), (0, left_pad),(0,0)), mode='constant', constant_values=0)

                disp_top_pad = max_h - dataL.shape[0]
                disp_left_pad = max_w - dataL.shape[1]
                # dataL = np.expand_dims(np.expand_dims(dataL, 0), 0)
                dataL = np.lib.pad(dataL, ((0, 0), (0, 0), (disp_top_pad, 0), (0, disp_left_pad)), mode='constant', constant_values=0)[0,0]
                dataL = np.ascontiguousarray(dataL, dtype=np.float32)

            processed = preprocess.get_transform()
            left_img = processed(left_img.float())
            right_img = processed(right_img.float())

            return (left_img, right_img, dataL)

        ##* validation (i.e. no data augmentation, only scailing and padding) *##
        if self.is_validation:
            left_img = np.array(left_img)

            right_img = np.array(right_img)
            left_img = cv2.resize(left_img,None,fx=self.testres,fy=self.testres,interpolation=cv2.INTER_CUBIC)
            right_img = cv2.resize(right_img,None,fx=self.testres,fy=self.testres,interpolation=cv2.INTER_CUBIC)
            left_img = self.processed(left_img).numpy()
            right_img = self.processed(right_img).numpy()

            left_img = np.reshape(left_img,[1,3,left_img.shape[1],left_img.shape[2]])
            right_img = np.reshape(right_img,[1,3,right_img.shape[1],right_img.shape[2]])

            ##fast pad
            max_h = int(left_img.shape[2] // 128 * 128)
            max_w = int(left_img.shape[3] // 128 * 128)
            if max_h < left_img.shape[2]: max_h += 128
            if max_w < left_img.shape[3]: max_w += 128

            top_pad = max_h-left_img.shape[2]
            left_pad = max_w-left_img.shape[3]
            left_img = np.lib.pad(left_img,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
            right_img = np.lib.pad(right_img,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
            left_img = left_img[0]
            right_img = right_img[0]
            return (left_img, right_img, dataL)

    def __len__(self):
        return len(self.left)
