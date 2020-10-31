#!/bin/sh
export PYTHONPATH=/root/high-res-stereo:$PYTHONPATH
name="model-final768_testres-1_maxdisp-2056"
for filename in /data/private/Argoverse/argoverse-tracking/train2/*; do
    python "/root/high-res-stereo/unlabeled_util/argoverse_make_pseudo_gt.py" "--name" "${name}" "--loadmodel" "./final-768px.tar"  "--testres" "1" "--datapath" "${filename}"
done

name="model-final768_testres-1_maxdisp-768"
for filename in /data/private/Argoverse/argoverse-tracking/train3/*; do
    python "/root/high-res-stereo/unlabeled_util/argoverse_make_pseudo_gt.py" "--name" "${name}" "--loadmodel" "./final-768px.tar"  "--testres" "1" "--datapath" "${filename}" "--max_disp" "768"
done

name="model-final768_testres-0.7_maxdisp-768"
for filename in /data/private/Argoverse/argoverse-tracking/train4/*; do
    python "/root/high-res-stereo/unlabeled_util/argoverse_make_pseudo_gt.py" "--name" "${name}" "--loadmodel" "./final-768px.tar"  "--testres" "0.7" "--datapath" "${filename}" "--max_disp" "768"
done