#!/bin/sh
CUDA_VISIBLE_DEVICES=0 \


#python3 source_pretrain.py -train dukemtmc_all -dt market1501  --height 256 --width 128 -ds dukemtmc -a resnet_ibn50a --adam --seed 0 --margin 0.0 \
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 100 \
#	--eval-step 10 --logs-dir logs/gan_expr/duke2market/ResNet_spgan_all_ibn

#python3 source_pretrain.py -train market1501_all -dt dukemtmc  --height 256 --width 128 -ds market1501 -a resnet_ibn50a --adam --seed 0 --margin 0.0 \
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 100 \
#	--eval-step 10 --logs-dir logs/gan_expr/market2duke/ResNet_spgan_all_ibn

#python3 source_pretrain.py -train dukemtmc_spgan_random -dt market1501  --height 256 --width 128 -ds dukemtmc -a resnet_ibn50a --adam --seed 0 --margin 0.0 \
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 100 \
#	--eval-step 10 --logs-dir logs/gan_expr/duke2market/ResNet_spgan_random_ibn

#python3 source_pretrain.py -train market1501_spgan_random -dt dukemtmc  --height 256 --width 128 -ds market1501 -a resnet_ibn50a --adam --seed 0 --margin 0.0 \
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 100 \
#	--eval-step 10 --logs-dir logs/gan_expr/market2duke/ResNet_spgan_random_ibn

python3 source_pretrain.py -train dukemtmc_spgan_all -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/gan_expr/duke2market/ResNet50_spgan_all_200_2

python3 source_pretrain.py -train market1501_spgan_all -dt dukemtmc  --height 256 --width 128 -ds market1501 -a ResNet50 --adam --seed 0 --margin 0.0 \
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/gan_expr/market2duke/ResNet50_spgan_all_200_2

python3 source_pretrain.py -train dukemtmc_spgan_random -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/gan_expr/duke2market/ResNet50_spgan_random_200_2

python3 source_pretrain.py -train market1501_spgan_random -dt dukemtmc  --height 256 --width 128 -ds market1501 -a ResNet50 --adam --seed 0 --margin 0.0 \
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/gan_expr/market2duke/ResNet50_spgan_random_200_2