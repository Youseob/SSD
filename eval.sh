#!/bin/bash
cd ~/DC

/home/sykim/anaconda3/envs/diffuser/bin/python eval.py --cid $1 --pid $2 --dataset halfcheetah-medium-expert-v2 \
                                                           --control every \
                                                           --increasing_condition True 
