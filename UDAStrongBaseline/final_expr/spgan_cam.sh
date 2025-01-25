#!/bin/sh
CUDA_VISIBLE_DEVICES=0 \


python3 source_pretrain.py -train dukemtmc_spgan_allcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/final_expr/duke2market/ResNet50_spgan_allcam_0

#python3 source_pretrain.py -train market1501_spgan_allcam -dt dukemtmc  --height 256 --width 128 -ds market1501 -a ResNet50 --adam --seed 0 --margin 0.0 \
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
#	--eval-step 10 --logs-dir logs/final_expr/market2duke/ResNet50_spgan_allcamd_0

python3 source_pretrain.py -train dukemtmc_spgan_randomcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/final_expr/duke2market/ResNet50_spgan_randomcam_0

#python3 source_pretrain.py -train market1501_spgan_allcam -dt dukemtmc  --height 256 --width 128 -ds market1501 -a ResNet50 --adam --seed 0 --margin 0.0 \
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
#	--eval-step 10 --logs-dir logs/final_expr/market2duke/ResNet50_spgan_randomcam_0

python3 source_pretrain.py -train dukemtmc_spgan_allcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/final_expr/duke2market/ResNet50_spgan_allcam_1

#python3 source_pretrain.py -train market1501_spgan_allcam -dt dukemtmc  --height 256 --width 128 -ds market1501 -a ResNet50 --adam --seed 0 --margin 0.0 \
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
#	--eval-step 10 --logs-dir logs/final_expr/market2duke/ResNet50_spgan_allcamd_1

python3 source_pretrain.py -train dukemtmc_spgan_randomcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/final_expr/duke2market/ResNet50_spgan_randomcam_1

#python3 source_pretrain.py -train market1501_spgan_allcam -dt dukemtmc  --height 256 --width 128 -ds market1501 -a ResNet50 --adam --seed 0 --margin 0.0 \
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
#	--eval-step 10 --logs-dir logs/final_expr/market2duke/ResNet50_spgan_randomcam_1