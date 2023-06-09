#!/bin/bash
cd ~/DC

/home/sykim/anaconda3/envs/diffuser/bin/python eval.py --cid $1 --pid $2 --dataset FetchPickAndPlace-v1 \
                                                           --control fetch \
                                                           --increasing_condition False
