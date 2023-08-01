#!/bin/bash
cd ~/DC
for increasing_condition in True False
do
for multi in True False
do
for _ in {0..9}
do
for seed in 3 4
do
    /home/sykim/anaconda3/envs/diffuser/bin/python eval.py --dataset maze2d-umaze-v1 \
                                                           --control position \
                                                           --increasing_condition $increasing_condition \
                                                           --multi $multi \
                                                           --target_v 1.0 \
                                                           --seed $seed \
                                                           --diffusion_epoch 999999
done
done
done
done