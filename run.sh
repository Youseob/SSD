#!/bin/bash
cd ~/DC
for diffusion_epoch in 9999 19999 29999 39999 49999 
do
for epi_seed in {0..9}
do
    /home/sykim/anaconda3/envs/diffuser/bin/python eval.py --condition_dropout 0.5 --diffusion_epoch $diffusion_epoch --epi_seed $epi_seed
done
done