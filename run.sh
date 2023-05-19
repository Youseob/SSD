#!/bin/bash
cd ~/DC
for diffusion_epoch in 99999
do
for epi_seed in {0..9}
do
    /home/sykim/anaconda3/envs/diffuser/bin/python eval.py --dataset walker2d-medium-expert-v2 \
                                                           --control every \
                                                           --decreasing_target False \
                                                            --diffusion_epoch $diffusion_epoch --epi_seed $epi_seed
done
done