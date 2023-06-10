#!/bin/bash
cd ~/DC
for diffusion_epoch in 999999
do
for epi_seed in {5..9}
do
for seed in {4..7}
do
    /home/sykim/anaconda3/envs/diffuser/bin/python eval.py --dataset FetchReach-v1 \
                                                           --control fetch \
                                                           --increasing_condition False \
                                                           --horizon 16 \
                                                           --seed $seed \
                                                            --diffusion_epoch $diffusion_epoch --epi_seed $epi_seed
done
done
done
