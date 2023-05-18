#!/bin/bash
cd ~/DC
for diffusion_epoch in 99999 89999 79999 69999
do
for epi_seed in {0..9}
do
    /home/sykim/anaconda3/envs/diffuser/bin/python eval.py --dataset maze2d-umaze-v1 \
                                                           --multi False \
                                                           --control torqueshort \
                                                            --diffusion_epoch $diffusion_epoch --epi_seed $epi_seed
done
done