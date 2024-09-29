#!/usr/bin/env bash
set -x
DATAPATH=/mnt/win/DataSet/
LOGDIR=./logs/test
MODEL=logs/trained/KT15.ckpt
python test.py --dataset kitti \
    --datapath $DATAPATH --testlist filenames/kt2015_testing.txt \
    --logdir $LOGDIR \
    --resume  $MODEL\
    --summary_freq 20 \
    --test_batch_size 1 \
    # --save_disp_to_file