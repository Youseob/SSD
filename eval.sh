#!/bin/bash
cd ~/DC

/home/sykim/anaconda3/envs/diffuser/bin/python eval.py --cid $1 --pid $2 --dataset maze2d-medium-v1 \
                                                           --control position \
                                                           --increasing_condition False \
