#!/bin/bash

# Train StarGAN using the CelebA dataset
python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 14 --parsing --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_seg 0 --lambda_idt 0\
               --sample_dir checkpoints/personreid_parsing_0_0/samples --log_dir checkpoints/personreid_parsing_0_0/logs \
               --model_save_dir checkpoints/personreid_parsing_0_0/models --result_dir checkpoints/personreid_parsing_0_0/results
#python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 14 --parsing --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_seg 0 --lambda_idt 1\
#               --sample_dir checkpoints/personreid_parsing_0_1/samples --log_dir checkpoints/personreid_parsing_0_1/logs \
#               --model_save_dir checkpoints/personreid_parsing_0_1/models --result_dir checkpoints/personreid_parsing_0_1/results
#python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 14 --parsing --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_seg 0 --lambda_idt 2\
#               --sample_dir checkpoints/personreid_parsing_0_2/samples --log_dir checkpoints/personreid_parsing_0_2/logs \
#               --model_save_dir checkpoints/personreid_parsing_0_2/models --result_dir checkpoints/personreid_parsing_0_2/results
#python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 14 --parsing --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_seg 0 --lambda_idt 5\
#               --sample_dir checkpoints/personreid_parsing_0_5/samples --log_dir checkpoints/personreid_parsing_0_5/logs \
#               --model_save_dir checkpoints/personreid_parsing_0_5/models --result_dir checkpoints/personreid_parsing_0_5/results
#
#python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 14 --parsing --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_seg 1 --lambda_idt 0\
#               --sample_dir checkpoints/personreid_parsing_1_0/samples --log_dir checkpoints/personreid_parsing_1_0/logs \
#               --model_save_dir checkpoints/personreid_parsing_1_0/models --result_dir checkpoints/personreid_parsing_1_0/results
#python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 14 --parsing --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_seg 1 --lambda_idt 1\
#               --sample_dir checkpoints/personreid_parsing_1_1/samples --log_dir checkpoints/personreid_parsing_1_1/logs \
#               --model_save_dir checkpoints/personreid_parsing_1_1/models --result_dir checkpoints/personreid_parsing_1_1/results
#python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 14 --parsing --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_seg 1 --lambda_idt 2\
#               --sample_dir checkpoints/personreid_parsing_1_2/samples --log_dir checkpoints/personreid_parsing_1_2/logs \
#               --model_save_dir checkpoints/personreid_parsing_1_2/models --result_dir checkpoints/personreid_parsing_1_2/results
#python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 14 --parsing --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_seg 1 --lambda_idt 5\
#               --sample_dir checkpoints/personreid_parsing_1_5/samples --log_dir checkpoints/personreid_parsing_1_5/logs \
#               --model_save_dir checkpoints/personreid_parsing_1_5/models --result_dir checkpoints/personreid_parsing_1_5/results
#
#
#python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 14 --parsing --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_seg 2 --lambda_idt 0\
#               --sample_dir checkpoints/personreid_parsing_2_0/samples --log_dir checkpoints/personreid_parsing_2_0/logs \
#               --model_save_dir checkpoints/personreid_parsing_2_0/models --result_dir checkpoints/personreid_parsing_2_0/results 
python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 14 --parsing --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_seg 2 --lambda_idt 1\
               --sample_dir checkpoints/personreid_parsing_2_1/samples --log_dir checkpoints/personreid_parsing_2_1/logs \
               --model_save_dir checkpoints/personreid_parsing_2_1/models --result_dir checkpoints/personreid_parsing_2_1/results 
python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 14 --parsing --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_seg 2 --lambda_idt 2\
               --sample_dir checkpoints/personreid_parsing_2_2/samples --log_dir checkpoints/personreid_parsing_2_2/logs \
               --model_save_dir checkpoints/personreid_parsing_2_2/models --result_dir checkpoints/personreid_parsing_2_2/results 
python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 14 --parsing --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_seg 2 --lambda_idt 5\
               --sample_dir checkpoints/personreid_parsing_2_5/samples --log_dir checkpoints/personreid_parsing_2_5/logs \
               --model_save_dir checkpoints/personreid_parsing_2_5/models --result_dir checkpoints/personreid_parsing_2_5/results 

python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 14 --parsing --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_seg 5 --lambda_idt 0\
               --sample_dir checkpoints/personreid_parsing_5_0/samples --log_dir checkpoints/personreid_parsing_5_0/logs \
               --model_save_dir checkpoints/personreid_parsing_5_0/models --result_dir checkpoints/personreid_parsing_5_0/results 
python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 14 --parsing --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_seg 5 --lambda_idt 1\
               --sample_dir checkpoints/personreid_parsing_5_1/samples --log_dir checkpoints/personreid_parsing_5_1/logs \
               --model_save_dir checkpoints/personreid_parsing_5_1/models --result_dir checkpoints/personreid_parsing_5_1/results 
python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 14 --parsing --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_seg 5 --lambda_idt 2\
               --sample_dir checkpoints/personreid_parsing_5_2/samples --log_dir checkpoints/personreid_parsing_5_2/logs \
               --model_save_dir checkpoints/personreid_parsing_5_2/models --result_dir checkpoints/personreid_parsing_5_2/results 
python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 14 --parsing --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_seg 5 --lambda_idt 5\
               --sample_dir checkpoints/personreid_parsing_5_5/samples --log_dir checkpoints/personreid_parsing_5_5/logs \
               --model_save_dir checkpoints/personreid_parsing_5_5/models --result_dir checkpoints/personreid_parsing_5_5/results 

python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 2 --parsing --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_seg 0 --lambda_idt 0 --cam single\
               --sample_dir checkpoints/personreid_parsing_singlecam/samples --log_dir checkpoints/personreid_parsing_singlecam/logs \
               --model_save_dir checkpoints/personreid_parsing_singlecam/models --result_dir checkpoints/personreid_parsing_singlecam/results
