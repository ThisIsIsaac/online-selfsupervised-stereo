#!/bin/sh
export PYTHONPATH=/root/high-res-stereo:$PYTHONPATH
name="model-final768_testres-0.5_maxdisp-1028"
for filename in /data/private/argoverse-tracking/train1/*; do
    python "/root/high-res-stereo/unlabeled_util/argoverse_make_pseudo_gt.py" "--name" "${name}" "--loadmodel" "./final-768px.tar"  "--testres" "0.5" "--datapath" "${filename}" "--max_disp" "1028"
done
