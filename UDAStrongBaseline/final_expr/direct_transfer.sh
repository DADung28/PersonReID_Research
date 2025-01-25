#!/bin/sh
CUDA_VISIBLE_DEVICES=0 \



#python3 source_pretrain.py -train dukemtmc -dt market1501  --height 256 --width 128 -ds dukemtmc -a resnet_ibn50a --adam --seed 0 --margin 0.0 \
#  --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 100 \
#	--eval-step 10 --logs-dir logs/gan_expr/duke2market/ResNet_ibn

#python3 source_pretrain.py -train market1501 -dt dukemtmc  --height 256 --width 128 -ds market1501 -a resnet_ibn50a --adam --seed 0 --margin 0.0 \
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 100 \
#	--eval-step 10 --logs-dir logs/gan_expr/market2duke/ResNet_ibn

python3 source_pretrain.py -train dukemtmc -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/final_expr/duke2market/ResNet50_0

python3 source_pretrain.py -train market1501 -dt dukemtmc  --height 256 --width 128 -ds market1501 -a ResNet50 --adam --seed 0 --margin 0.0 \
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/final_expr/market2duke/ResNet50_0

python3 source_pretrain.py -train dukemtmc -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/final_expr/duke2market/ResNet50_1

python3 source_pretrain.py -train market1501 -dt dukemtmc  --height 256 --width 128 -ds market1501 -a ResNet50 --adam --seed 0 --margin 0.0 \
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/final_expr/market2duke/ResNet50_1