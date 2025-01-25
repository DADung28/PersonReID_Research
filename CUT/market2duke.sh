#!/bin/bash 


python3 copilot_train.py --model copilot --dataroot /home/jun/ReID_Dataset/ --name market2duke_copilot --print_freq 500 --image_save_dir './sample_images/market2duke/copilot' --dataset_mode labeled --serial_batches --direction AtoB
python3 copilot_train.py --model copilot --dataroot /home/jun/ReID_Dataset/ --name market2duke_copilot --print_freq 500 --image_save_dir './sample_images/market2duke/copilot' --dataset_mode labeled --serial_batches --direction AtoB
#python3 train.py --model segcut --dataroot /home/jun/CUT/datasets/market2duke --name market2duke_segcut --print_freq 1000 --image_save_dir './sample_images/market2duke/cut_segemetation' 


#python3 train.py --dataroot /home/jun/CUT/datasets/market2duke --name market2duke
#python3 test.py --dataroot /home/jun/CUT/datasets/market2duke --name market2duke --results_dir ./results
