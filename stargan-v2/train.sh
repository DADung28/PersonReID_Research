#!/bin/bash

python main.py --mode train --num_domains 14 --w_hpf 0 --batch_size 8\
               --lambda_reg 1 --lambda_sty 1 --lambda_ds 2 --lambda_cyc 1 \
               --train_img_dir /home/jun/stargan-v2/data/personreid/train \
               --val_img_dir /home/jun/stargan-v2/data/personreid/val \
               --sample_dir /home/jun/stargan-v2/jikken/personreid/samples \
               --checkpoint_dir /home/jun/stargan-v2/jikken/personreid/checkpoints 


