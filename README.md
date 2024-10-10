# Pseudo-Stereo Inputs: A Solution to the Occlusion Challenge in Self-Supervised Stereo Matching

The code for ["Pseudo-Stereo Inputs: A Solution to the Occlusion Challenge in Self-Supervised Stereo Matching"](https://arxiv.org/abs/2410.02534).

## Citation

## Results
![kt15-result](assets/KT4.png)
## Settings

The code is tested with Pytorch 2.2.1, CUDA 12, Python 3.11 and Ubuntu 22.04 using a single GPU. The requirements.txt is provided.

```sh
conda create -n pseudo-stereo python=3.11
conda activate pseudo-stereo
pip install -r requirements
```

## Train

A train.sh file is provided for KITTI dataset. The `DATAPATH` in train.sh needs to be replaced to your path of KITTI dataset.

```sh
DATAPATH=YOUR_DATAPATH
LOGDIR=./logs
python main.py \
    --datapath $DATAPATH \
    --logdir $LOGDIR \
    --batch_size 8 \
    --test_batch_size 6 \
    --lr 0.0001 \
    --maxdisp 192 \
    --message="message" \
    --all_fake=0 \
    --occ_detect \
    --dataset kitti \
    --trainlist filenames/kt2015_all_repeat.txt \
    --testlist filenames/kt2015_val.txt

```

## Test

Pre-trained model for KITTI 2012 / KITTI 2015 is available at [Google Drive](https://drive.google.com/drive/folders/1NBj6ZqL33ZL2-OJJne_AETu6Z6PxvzqU?usp=sharing).

A test.sh file is provided for KITTI dataset. The `DATAPATH` needs to be replaced to your path of KITTI dataset and the `MODEL` needs to be set as the path of trained model.

```sh
DATAPATH=YOUR_DATAPATH
LOGDIR=./logs/test
MODEL=logs/trained/KT15.ckpt
python test.py --dataset kitti \
    --datapath $DATAPATH --testlist filenames/kt2015_testing.txt \
    --logdir $LOGDIR \
    --loadckpt  $MODEL\
    --summary_freq 20 \
    --test_batch_size 1 \
    # --save_disp_to_file
```

## Acknowledgment

We would like to express our gratitude to the following papers and repositories.

[1] [PSMNet](https://github.com/JiaRenChang/PSMNet/tree/master): [Pyramid Stereo Matching Network](https://arxiv.org/abs/1803.08669)

[2] [Monodepth2](https://github.com/nianticlabs/monodepth2): [Digging into Self-Supervised Monocular Depth Prediction.](https://arxiv.org/abs/1806.01260)
