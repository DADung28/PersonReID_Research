#!/bin/sh
CUDA_VISIBLE_DEVICES=0 \


#python3 last_pretrain.py -train last -ds market1501  --height 256 --width 128 -dt dukemtmc -a resnet_ibn50a --adam --seed 0 --margin 0.0 \
#  --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 1100 --epochs 100 \
#   --eval-step 10 --logs-dir logs/jikken/market2duke/ResNetIbn_200_0.00035

#python3 last_pretrain.py -train last -ds market1501  --height 256 --width 128 -dt dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 1100 --epochs 100 \
#   --eval-step 10 --logs-dir logs/jikken/last/ResNet_200_0.00035

python3 last_pretrain.py -train last -ds market1501 --height 224 --width 224 -dt dukemtmc -a Swin --seed 0 --margin 0.0 \
	--num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 100 \
	--eval-step 10 --logs-dir logs/jikken/last/Swin_200_0.00035

python3 last_pretrain.py -train last -ds market1501 --height 224 --width 224 -dt dukemtmc -a Swin --seed 0 --margin 0.0 \
	--num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.001 --milestones 40 70 --iters 200 --epochs 100 \
	--eval-step 10 --logs-dir logs/jikken/last/Swin_200_0.001

python3 last_pretrain.py -train last -ds market1501 --height 224 --width 224 -dt dukemtmc -a ViT --seed 0 --margin 0.0 \
	--num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 100 \
	--eval-step 10 --logs-dir logs/jikken/last/ViT_200_0.00035

python3 last_pretrain.py -train last -ds market1501 --height 224 --width 224 -dt dukemtmc -a ViT --seed 0 --margin 0.0 \
	--num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.001 --milestones 40 70 --iters 200 --epochs 100 \
	--eval-step 10 --logs-dir logs/jikken/last/ViT_200_0.001	