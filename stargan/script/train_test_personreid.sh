#!/bin/bash

# Train StarGAN using the CelebA dataset
python main.py --mode train --dataset PersonReID --image_size 286 --c_dim 14 --batch_size 8 --num_iters 300000 --num_iters_decay 200000  --device 0 \
               --sample_dir checkpoints/personreid/samples --log_dir checkpoints/personreid/logs \
               --model_save_dir checkpoints/personreid/models --result_dir checkpoints/personreid/results \
               

# Test StarGAN using the CelebA dataset
#python main.py --mode test --dataset CelebA --image_size 128 --c_dim 5 \
#               --sample_dir stargan_celeba/samples --log_dir stargan_celeba/logs \
#               --model_save_dir stargan_celeba/models --result_dir stargan_celeba/results \
#              --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young

