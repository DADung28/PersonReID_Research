#!/bin/bash

#python3 my_train.py --model cut --name market2duke_original --direction AtoB \
#        --dataroot /home/jun/ReID_Dataset --image_save_dir sample_images/market2duke/original \
#        --print_freq 100 --n_epochs 15 --n_epochs_decay 5 --batch_size 4 --iters 3000

#python3 my_train.py --model cut --name duke2market_original --direction BtoA \
#        --dataroot /home/jun/ReID_Dataset --image_save_dir sample_images/duke2market/original\
#        --print_freq 100 --n_epochs 15 --n_epochs_decay 5 --batch_size 4 --iters 3000

python3 data_transfer.py --model cut --name market2duke_original --direction AtoB \
        --source_dir /home/jun/ReID_Dataset/Market-1501-v15.09.15 \
        --save_dir /home/jun/ReID_Dataset/Market-1501-v15.09.15-cut

python3 data_transfer.py --model cut --name duke2market_original --direction BtoA \
        --source_dir /home/jun/ReID_Dataset/DukeMTMC-reID \
        --save_dir /home/jun/ReID_Dataset/DukeMTMC-reID-cut