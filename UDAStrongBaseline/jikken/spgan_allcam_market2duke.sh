#!/bin/sh
CUDA_VISIBLE_DEVICES=0 \

#python3 source_pretrain.py -train market1501_spgan_allcam -ds market1501  --height 256 --width 128 -dt dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 1800 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/market2duke/ResNet_spgan_allcam_1800_0.00035


#thon3 source_pretrain.py -train market1501_spgan_allcam -ds market1501  --height 256 --width 128 -dt dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/market2duke/ResNet_spgan_allcam_200_0.00035


python3 source_pretrain.py -train market1501_spgan_randomcam -ds market1501  --height 256 --width 128 -dt dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 100 \
	--eval-step 10 --logs-dir logs/jikken/market2duke/ResNet_spgan_randomcam_new_200_0.00035

#python3 source_pretrain.py -train market1501_spgan_random -ds market1501  --height 256 --width 128 -dt dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/market2duke/ResNet_spgan_random_200_0.00035

#python3 source_pretrain.py -train market1501_spgan_randomcam -ds market1501  --height 224 --width 224 -dt dukemtmc -a Swin --seed 0 --margin 0.0 \
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/market2duke/Swin_spgan_randomcam_200_0.00035

#python3 source_pretrain.py -train market1501_spgan_random -ds market1501  --height 224 --width 224 -dt dukemtmc -a Swin --seed 0 --margin 0.0 \
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.001 --milestones 40 70 --iters 200 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/market2duke/_spgan_random_200_0.001