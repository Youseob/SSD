#!/bin/bash
cd ~/DC
for target_v in 0.025 0.05 0.1 0.2 0.4 0.8 1.0
do
for multi in True
do
for _ in {0..9}
do
for seed in 6
do
    /home/sykim/anaconda3/envs/diffuser/bin/python eval.py --dataset FetchSlide-v1 \
                                                           --control every \
                                                           --increasing_condition False \
                                                           --target_v $target_v \
                                                           --seed $seed \
                                                           --diffusion_epoch 1199999 \
                                                        #    --multi $multi 
done
done
done
done