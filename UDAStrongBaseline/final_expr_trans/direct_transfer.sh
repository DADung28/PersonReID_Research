#!/bin/sh
CUDA_VISIBLE_DEVICES=0 \

python3 source_pretrain.py -train dukemtmc -dt market1501 --height 224 --width 224 -ds dukemtmc -a Swin --seed 0 --margin 0.0 \
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 100 \
	--eval-step 10 --logs-dir logs/final_expr_trans/duke2market/Swin_0

python3 source_pretrain.py -train market1501 -dt dukemtmc --height 224 --width 224 -ds market1501 -a Swin --seed 0 --margin 0.0 \
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 100 \
	--eval-step 10 --logs-dir logs/final_expr_trans/market2duke/Swin_0

#python3 source_pretrain.py -train dukemtmc -dt market1501  --height 224 --width 224 -ds dukemtmc -a Swin --seed 0 --margin 0.0 \
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 100 \
#	--eval-step 10 --logs-dir logs/final_expr_trans/duke2market/Swin_1

#python3 source_pretrain.py -train market1501 -d:t dukemtmc  --height 224 --width 224 -ds market1501 -a Swin --seed 0 --margin 0.0 \
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 100 \
#	--eval-step 10 --logs-dir logs/final_expr_trans/market2duke/Swin_1