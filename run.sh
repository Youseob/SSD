#!/bin/bash
cd ~/DC
for diffusion_epoch in 399999 499999
do
for _ in {0..9}
do
for seed in {0..8}
do
for seed in {4..7}
do
    /home/sykim/anaconda3/envs/diffuser/bin/python eval.py --dataset FetchReach-v1 \
                                                           --control fetch \
                                                           --increasing_condition False \
                                                           --horizon 16 \
                                                           --seed $seed \
                                                            --diffusion_epoch $diffusion_epoch
done
done
done
done
