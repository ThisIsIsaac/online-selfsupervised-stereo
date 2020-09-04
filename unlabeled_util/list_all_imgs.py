import os
import random
from click.exceptions import FileError

if __name__ == "__main__":
    root_path = "/DATA1/isaac/"
    
    dirs = os.walk(os.path.join(root_path, "KITTI_raw"))
    
    img_paths = []
    
    for (dirpath, dirnames, filenames) in dirs:
        if "image_02/data" in dirpath:
            if len(dirnames) != 0:
                raise FileError
            start_idx = dirpath.find("KITTI_raw")
            img_dir = dirpath[start_idx:]
            for imgname in filenames:

                img_paths.append(os.path.join(img_dir, imgname))
            # print("dirpath: " + str(dirpath))
            # print("dirnames:" + str(dirnames))
            # print("filenames:" + str(filenames))
    
    img_paths.sort()
    
    # with open(os.path.join(root_path, "KITTI_raw", "all_img_paths.txt"), "w") as file:
    #     for path in img_paths:
    #         file.write(path + "\n")
    
    exp_train_set = random.sample(img_paths, len(img_paths)//10)
    with open(os.path.join(root_path, "KITTI_raw", "exp_train_set.txt"), "w") as file:
        for path in exp_train_set:
            file.write(path + "\n")

