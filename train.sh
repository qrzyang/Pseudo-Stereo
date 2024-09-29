#!/usr/bin/env bash
set -x
DATAPATH=/mnt/win/DataSet/
LOGDIR=./logs/fake_psm
python main.py \
    --datapath $DATAPATH \
    --logdir $LOGDIR \
    --batch_size 8 \
    --test_batch_size 6 \
    --lr 0.0001 \
    --maxdisp 192 \
    --message="flip + occ + all_fake, smooth 0.5, kt15, wider" \
    --all_fake=0 \
    --occ_detect \
    --dataset kitti \
    --trainlist filenames/kt2015_all_repeat.txt \
    --testlist filenames/kt2015_val.txt
