import os
from os import mkdir
from shutil import copyfile
import time

def prepare_kitti(in_path, obj_path, out_path):
    if not (os.path.isabs(in_path) and os.path.isabs(out_path) and os.path.isabs(obj_path)):
        raise ValueError("in_path, obj_path, out_path must be absolute paths")

    if not os.path.exists(in_path):
        raise FileNotFoundError(str(in_path) + " doesn't exist")

    if not os.path.exists(out_path):
        raise FileNotFoundError(out_path + " doesn't exist")
    
    if not os.path.exists(obj_path):
        raise FileNotFoundError(obj_path + " doesn't exist")

    folder_name = time.strftime("kitti_%x_%X")
    folder_name = folder_name.replace("/", "-")
    folder_name = folder_name.replace(":", "-")
    out_path = os.path.join(out_path, folder_name)
    os.mkdir(out_path)

    new_disp_dir = os.path.join(out_path, "disp_0")
    os.mkdir(new_disp_dir)

    new_obj_dir = os.path.join(out_path, "obj_map")
    os.mkdir(new_obj_dir)

    for dir in os.listdir(in_path):
        if not dir.lower().startswith("kitti"): continue
        disp_path = os.path.join(in_path, dir, "disp.png")

        if not os.path.exists(disp_path):
            raise FileNotFoundError(str(disp_path) + " doesn't exist")

        new_disp_name = dir[-9:] + ".png"
        new_disp_path = os.path.join(out_path, new_disp_dir, new_disp_name)

        

        copyfile(disp_path, new_disp_path)
        
    for dir in os.listdir(obj_path):
        if not dir.lower().startswith("kitti"): continue
        old_obj_path = os.path.join(obj_path, dir, "obj_map.png")

        if not os.path.exists(old_obj_path):
            raise FileNotFoundError(str(old_obj_path) + " doesn't exist")
            
        new_obj_name = dir[-9:] + ".png"
        new_obj_path = os.path.join(out_path, new_obj_dir, new_obj_name)
        
        copyfile(old_obj_path, new_obj_path)
    
        
if __name__ == "__main__":
    in_path = "/home/isaac/high-res-stereo/output/eval_kitti_model_color"
    obj_path = "/home/isaac/rvc_devkit/stereo/datasets_middlebury2014/training"
    out_path = "/home/isaac/high-res-stereo/kitti_submission_output"
    prepare_kitti(in_path,obj_path,  out_path)