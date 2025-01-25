#!/bin/bash 

python3 train.py --model segcut --dataroot /home/jun/GAN_dataset/duke2market --name duke2market_segcut --print_freq 1000 --image_save_dir './sample_images/duke2market/cut_segemetation_v2' --gpu_ids 2,3
#python3 test.py --dataroot /home/jun/CUT/datasets/duke2market --name duke2market --results_dir ./results
