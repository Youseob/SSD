#!/bin/bash
cd /home/sykim/DC

/home/sykim/anaconda3/envs/diffuser/bin/python train.py --dataset maze2d-umaze-v1 --max_path_length 250
                                            