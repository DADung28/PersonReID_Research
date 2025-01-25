#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 sbs_traindbscan_unc.py -tt dukemtmc -st market1501 -a resnet50_multi\
	--num-instances 4 --lr 0.00035 --iters 200 -b 64 --epochs 100 \
	--dropout 0 --n-jobs 16 --choice_c 0 \
	--init-1 logs/market1501TOdukemtmc/ResNet50_original_multi/model_best.pth.tar \
	--logs-dir logs/dbscan-market1501TOdukemtmc/ResNet50_original_uncertainly_multi

# DBSCAN uncertantly
python3 sbs_traindbscan_unc.py --height 256 --width 128  --lr 0.001  -tt dukemtmc -st market1501 -a ResNet50\
	--num-instances 4 --iters 200 -b 64 --epochs 100 \
	--dropout 0 --n-jobs 16 --choice_c 0 \
	--init-1 logs/market1501TOdukemtmc/market_pretrain_resnet/model_best.pth.tar \
	--logs-dir logs/dbscan-market1501TOdukemtmc/ResNet50_uncertainly