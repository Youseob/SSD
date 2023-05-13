#!/bin/bash
cd ~/DC
for diffusion_epoch in 19999
do
for epi_seed in {0..9}
do
    /home/sykim/anaconda3/envs/diffuser/bin/python eval.py --maxq True --diffusion_epoch $diffusion_epoch --epi_seed $epi_seed
done
done