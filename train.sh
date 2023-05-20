#!/bin/bash
cd /home/sykim/DC

/home/sykim/anaconda3/envs/diffuser/bin/python train.py --dataset walker2d-medium-expert-v2 --max_path_length 1000
                                            