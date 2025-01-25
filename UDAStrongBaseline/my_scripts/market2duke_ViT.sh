#!/bin/sh
CUDA_VISIBLE_DEVICES=0 \
python3 source_pretrain.py --height 224 --width 224 -ds market1501 -dt dukemtmc -a ViT --seed 0 --margin 0.0 \
	--num-instances 4 -b 32 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 400 --epochs 80 \
	--eval-step 20 --logs-dir logs/market1501TOdukemtmc/ViT

python3 sbs_traindbscan.py --height 224 --width 224 -tt dukemtmc -st market1501 -a ViT\
	--num-instances 4 --lr 0.00035 --iters 400 -b 32 --epochs 200 \
	--dropout 0 --n-jobs 16 --choice_c 0 \
	--init-1 logs/market1501TOdukemtmc/ViT/model_best.pth.tar \
	--logs-dir logs/dbscan-market1501TOdukemtmc/ViT
#python3 source_pretrain.py -ds market1501 -dt dukemtmc -a resnet50_sbs --seed 0 --margin 0.0 \
#	--num-instances 4 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 80 \
#	--eval-step 20 --logs-dir logs/market1501TOdukemtmc/resnet50_sbs-pretrain-0_norm

