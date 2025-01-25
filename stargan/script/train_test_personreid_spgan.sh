#!/bin/bash

# One cam is one domain
python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 14 --spgan --margin 1 --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_spgan 0 --lambda_idt 0\
               --sample_dir checkpoints/personreid_spgan_0_0/samples --log_dir checkpoints/personreid_spgan_0_0/logs \
               --model_save_dir checkpoints/personreid_spgan_0_0/models --result_dir checkpoints/personreid_spgan_0_0/results
python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 14 --spgan --margin 1 --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_spgan 1 --lambda_idt 0\
               --sample_dir checkpoints/personreid_spgan_1_0/samples --log_dir checkpoints/personreid_spgan_1_0/logs \
               --model_save_dir checkpoints/personreid_spgan_1_0/models --result_dir checkpoints/personreid_spgan_1_0/results
python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 14 --spgan --margin 1 --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_spgan 2 --lambda_idt 0\
               --sample_dir checkpoints/personreid_spgan_2_0/samples --log_dir checkpoints/personreid_spgan_2_0/logs \
               --model_save_dir checkpoints/personreid_spgan_2_0/models --result_dir checkpoints/personreid_spgan_2_0/results
python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 14 --spgan --margin 1 --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_spgan 0 --lambda_idt 5\
               --sample_dir checkpoints/personreid_spgan_0_5/samples --log_dir checkpoints/personreid_spgan_0_5/logs \
               --model_save_dir checkpoints/personreid_spgan_0_5/models --result_dir checkpoints/personreid_spgan_0_5/results
python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 14 --spgan --margin 1 --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_spgan 1 --lambda_idt 5\
               --sample_dir checkpoints/personreid_spgan_1_5/samples --log_dir checkpoints/personreid_spgan_1_5/logs \
               --model_save_dir checkpoints/personreid_spgan_1_5/models --result_dir checkpoints/personreid_spgan_1_5/results
python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 14 --spgan --margin 1 --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_spgan 2 --lambda_idt 5\
               --sample_dir checkpoints/personreid_spgan_2_5/samples --log_dir checkpoints/personreid_spgan_2_5/logs \
               --model_save_dir checkpoints/personreid_spgan_2_5/models --result_dir checkpoints/personreid_spgan_2_5/results

# All cam is one domain
python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 2 --spgan --margin 1 --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_spgan 0 --lambda_idt 0 --cam single\
               --sample_dir checkpoints/personreid_spgan_singlecam_0_0/samples --log_dir checkpoints/personreid_spgan_singlecam_0_0/logs \
               --model_save_dir checkpoints/personreid_spgan_singlecam_0_0/models --result_dir checkpoints/personreid_spgan_singlecam_0_0/results
python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 2 --spgan --margin 1 --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_spgan 1 --lambda_idt 0 --cam single\
               --sample_dir checkpoints/personreid_spgan_singlecam_1_0/samples --log_dir checkpoints/personreid_spgan_singlecam_1_0/logs \
               --model_save_dir checkpoints/personreid_spgan_singlecam_1_0/models --result_dir checkpoints/personreid_spgan_singlecam_1_0/results
python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 2 --spgan --margin 1 --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_spgan 2 --lambda_idt 0 --cam single\
               --sample_dir checkpoints/personreid_spgan_singlecam_2_0/samples --log_dir checkpoints/personreid_spgan_singlecam_2_0/logs \
               --model_save_dir checkpoints/personreid_spgan_singlecam_2_0/models --result_dir checkpoints/personreid_spgan_singlecam_2_0/results
python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 2 --spgan --margin 1 --batch_size 8 --num_iters 200000 --num_iters_decay 400000  --device 0 --lambda_spgan 0 --lambda_idt 5 --cam single\
               --sample_dir checkpoints/personreid_spgan_singlecam_0_5/samples --log_dir checkpoints/personreid_spgan_singlecam_0_5/logs \
               --model_save_dir checkpoints/personreid_spgan_singlecam_0_5/models --result_dir checkpoints/personreid_spgan_singlecam_0_5/results
python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 2 --spgan --margin 1 --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_spgan 1 --lambda_idt 5 --cam single\
               --sample_dir checkpoints/personreid_spgan_singlecam_1_5/samples --log_dir checkpoints/personreid_spgan_singlecam_1_5/logs \
               --model_save_dir checkpoints/personreid_spgan_singlecam_1_5/models --result_dir checkpoints/personreid_spgan_singlecam_1_5/results
python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 2 --spgan --margin 1 --batch_size 8 --num_iters 200000 --num_iters_decay 100000  --device 0 --lambda_spgan 2 --lambda_idt 5 --cam single\
               --sample_dir checkpoints/personreid_spgan_singlecam_2_5/samples --log_dir checkpoints/personreid_spgan_singlecam_2_5/logs \
               --model_save_dir checkpoints/personreid_spgan_singlecam_2_5/models --result_dir checkpoints/personreid_spgan_singlecam_2_5/results

