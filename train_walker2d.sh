#!/bin/bash
cd /home/sykim/DC
source ~/.bashrc

D4RL_DATASET_DIR=/tmp /home/sykim/anaconda3/envs/diffuser/bin/python3 train.py --cid $1 --pid $2 --dataset walker2d-medium-replay-v2 --max_path_length 1000 
#D4RL_DATASET_DIR=/tmp /home/sykim/anaconda3/envs/diffuser/bin/python3 train.py --dataset maze2d-large-v1 --max_path_length 600

                                            
