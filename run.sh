#!/bin/bash
cd ~/DC
for multi in True False
do
for _ in {0..9}
do
for seed in {0..2}
do
    /home/sykim/anaconda3/envs/diffuser/bin/python eval.py --dataset maze2d-umaze-v1 \
                                                           --control position \
                                                           --increasing_condition False \
                                                           --multi $multi \
                                                           --seed $seed 
done
done
done
