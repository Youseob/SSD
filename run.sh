#!/bin/bash
cd ~/DC
for target_v in 0.2
do
for multi in False
do
for _ in {0..9}
do
for seed in 3
do
    /home/sykim/anaconda3/envs/diffuser/bin/python eval.py --dataset maze2d-large-v1 \
                                                           --control position \
                                                           --increasing_condition False \
                                                           --multi $multi \
                                                           --target_v $target_v \
                                                           --seed $seed \
                                                           --diffusion_epoch 999999
done
done
done
done