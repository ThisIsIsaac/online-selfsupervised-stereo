import os

def get_kitti_raw_paths(root_dir, get_outdir=False, name=None):
    im0_paths = []
    im1_paths = []
    out_paths = []

    if get_outdir == True:
        if name == None:
            raise ValueError("give a name")

    for dir in os.listdir(root_dir):
        dir = os.path.join(root_dir, dir)

        if os.path.isdir(dir):
            left_path = os.path.join(os.path.join(dir, "image_02"), "data")
            right_path = os.path.join(os.path.join(dir, "image_03"), "data")

            assert(len(os.listdir(left_path)) == len(os.listdir(right_path)))

            for img in os.listdir(left_path):
                im0_paths.append(os.path.join(left_path, img))

            for img in os.listdir(right_path):
                im1_paths.append(os.path.join(right_path, img))

            if get_outdir == True:
                outdir = os.path.join(dir, name)
                if os.path.exists(outdir):
                    raise FileExistsError("Output folder with name = " + name + " already exists")

                os.mkdir(outdir)

                num_img = len(os.listdir(left_path))
                out_paths.extend([outdir] * num_img)

    return im0_paths, im1_paths, out_paths


