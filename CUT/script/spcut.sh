#!/bin/bash

#python3 my_train.py --model spcut --name market2duke_spcut --direction AtoB \
#        --dataroot /home/jun/ReID_Dataset --image_save_dir sample_images/market2duke/spcut \
#        --print_freq 100 --n_epochs 15 --n_epochs_decay 5 --batch_size 4 --iters 3000

#python3 my_train.py --model spcut --name duke2market_spcut --direction BtoA \
#        --dataroot /home/jun/ReID_Dataset --image_save_dir sample_images/duke2market/spcut\
#        --print_freq 100 --n_epochs 15 --n_epochs_decay 5 --batch_size 4 --iters 3000

python3 data_transfer.py --model spcut --name market2duke_spcut_lambda2 --direction AtoB \
        --source_dir /home/jun/ReID_Dataset/Market-1501-v15.09.15 \
        --save_dir /home/jun/ReID_Dataset/Market-1501-v15.09.15-spcut-ver2

python3 data_transfer.py --model spcut --name duke2market_spcut_lambda2 --direction BtoA \
        --source_dir /home/jun/ReID_Dataset/DukeMTMC-reID \
        --save_dir /home/jun/ReID_Dataset/DukeMTMC-reID-spcut-ver2