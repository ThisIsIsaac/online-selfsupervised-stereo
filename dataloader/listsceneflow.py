import torch.utils.data as data
import pickle
from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    ]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def scene_dataloader(filepath, val=False):

    classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
    image = [img for img in classes if img.find('cleanpass') > -1]
    disp  = [dsp for dsp in classes if dsp.find('disparity') > -1]

    monkaa_path = filepath + [x for x in image if 'monkaa' in x][0]
    monkaa_disp = filepath + [x for x in disp if 'monkaa' in x][0]

    monkaa_dir  = os.listdir(monkaa_path)

    all_left_img=[]
    all_right_img=[]
    all_left_disp = []
    all_right_disp = []
    test_left_img=[]
    test_right_img=[]
    test_left_disp = []
    test_right_disp = []


    for dd in monkaa_dir:
        for im in os.listdir(monkaa_path+'/'+dd+'/left/'):
            if is_image_file(monkaa_path+'/'+dd+'/left/'+im):
                all_left_img.append(monkaa_path+'/'+dd+'/left/'+im)
                all_left_disp.append(monkaa_disp+'/'+dd+'/left/'+im.split(".")[0]+'.pfm')
                all_right_disp.append(monkaa_disp+'/'+dd+'/right/'+im.split(".")[0]+'.pfm')

        for im in os.listdir(monkaa_path+'/'+dd+'/right/'):
            if is_image_file(monkaa_path+'/'+dd+'/right/'+im):
                all_right_img.append(monkaa_path+'/'+dd+'/right/'+im)

    flying_path = filepath + [x for x in image if x == 'flyingthings_cleanpass'][0]
    flying_disp = filepath + [x for x in disp if x == 'flyingthings_disparity'][0]
    flying_dir = flying_path+'/TRAIN/'
    subdir = ['A','B','C']

    for ss in subdir:
        flying = os.listdir(flying_dir+ss)

        for ff in flying:
            imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
            for im in imm_l:
                if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
                    all_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)

                all_left_disp.append(flying_disp+'/TRAIN/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')
                all_right_disp.append(flying_disp+'/TRAIN/'+ss+'/'+ff+'/right/'+im.split(".")[0]+'.pfm')

                if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
                    all_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)

    flying_dir = flying_path+'/TEST/'

    subdir = ['A','B','C']

    for ss in subdir:
        flying = os.listdir(flying_dir+ss)

        for ff in flying:
            imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
            for im in imm_l:
                if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
                    test_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)

                test_left_disp.append(flying_disp+'/TEST/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')
                test_right_disp.append(flying_disp+'/TEST/'+ss+'/'+ff+'/right/'+im.split(".")[0]+'.pfm')

                if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
                    test_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)


    driving_dir = filepath + [x for x in image if 'driving' in x][0] + '/'
    driving_disp = filepath + [x for x in disp if 'driving' in x][0]

    ##TODO was using 15 only
    subdir1 = ['35mm_focallength','15mm_focallength']
    subdir2 = ['scene_backwards','scene_forwards']
    subdir3 = ['fast','slow']


    for i in subdir1:
        for j in subdir2:
            for k in subdir3:
                imm_l = os.listdir(driving_dir+i+'/'+j+'/'+k+'/left/')
                for im in imm_l:
                    if is_image_file(driving_dir+i+'/'+j+'/'+k+'/left/'+im):
                        all_left_img.append(driving_dir+i+'/'+j+'/'+k+'/left/'+im)
                    all_left_disp.append(driving_disp+'/'+i+'/'+j+'/'+k+'/left/'+im.split(".")[0]+'.pfm')
                    all_right_disp.append(driving_disp+'/'+i+'/'+j+'/'+k+'/right/'+im.split(".")[0]+'.pfm')

                    if is_image_file(driving_dir+i+'/'+j+'/'+k+'/right/'+im):
                        all_right_img.append(driving_dir+i+'/'+j+'/'+k+'/right/'+im)

    all_left_img.sort()
    all_right_img.sort()
    all_left_disp.sort()
    all_right_disp.sort()

    with open("dataloader/scene_flow_val_idx.pkl", "rb") as file:
        val_idx_list = pickle.load(file)

    val_left_img = []
    val_right_img = []
    val_left_disp = []
    val_right_disp = []

    train_left_img = [all_left_img[idx] for idx in range(len(all_left_img)) if idx not in val_idx_list]
    train_right_img = [all_right_img[idx] for idx in range(len(all_left_img)) if idx not in val_idx_list]
    train_left_disp = [all_left_disp[idx] for idx in range(len(all_left_img)) if idx not in val_idx_list]
    train_right_disp = [all_right_disp[idx] for idx in range(len(all_left_img)) if idx not in val_idx_list]

    if val:
        val_left_img = [all_left_img[idx] for idx in val_idx_list]
        val_right_img = [all_right_img[idx] for idx in val_idx_list]
        val_left_disp = [all_left_disp[idx] for idx in val_idx_list]
        val_right_disp = [all_right_disp[idx] for idx in val_idx_list]




    return train_left_img, train_right_img, train_left_disp, train_right_disp, val_left_img, val_right_img, val_left_disp, val_right_disp


