#!/bin/bash
cd ~/DC
for diffusion_epoch in 549999
do
for epi_seed in {0..9}
do
    /home/sykim/anaconda3/envs/diffuser/bin/python eval.py --dataset maze2d-medium-v1 \
                                                           --control position \
                                                           --increasing_condition False \
                                                           --multi False \
                                                            --diffusion_epoch $diffusion_epoch --epi_seed $epi_seed
done
done