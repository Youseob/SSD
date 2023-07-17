#!/bin/bash
cd ~/DC
for diffusion_epoch in 499999
do
for _ in {0..9}
do
for seed in 0 3
do
    /home/sykim/anaconda3/envs/diffuser/bin/python eval.py --dataset maze2d-large-v1 \
                                                           --control position \
                                                           --increasing_condition False \
                                                           --multi True \
                                                           --seed $seed \
                                                            --diffusion_epoch $diffusion_epoch
done
done
done
