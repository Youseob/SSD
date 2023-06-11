#!/bin/bash
cd ~/DC
for diffusion_epoch in 99999
do
for _ in {0..9}
do
for seed in 0
do
    /home/sykim/anaconda3/envs/diffuser/bin/python eval.py --dataset FetchPickAndPlace-v1 \
                                                           --control fetch \
                                                           --increasing_condition False \
                                                           --horizon 32 \
                                                           --seed $seed \
                                                            --diffusion_epoch $diffusion_epoch
done
done
done
