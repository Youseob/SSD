#!/bin/bash
cd ~/DC
for diffusion_epoch in 699999
do
for epi_seed in {0..9}
do
    /home/sykim/anaconda3/envs/diffuser/bin/python eval.py --dataset FetchPickAndPlace-v1 \
                                                           --control fetch \
                                                           --increasing_condition False \
                                                           --horizon 20 \
                                                           --seed 0 \
                                                            --diffusion_epoch $diffusion_epoch --epi_seed $epi_seed
done
done