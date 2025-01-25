#!/bin/sh
CUDA_VISIBLE_DEVICES=0 \
python3 source_pretrain.py -dt market1501 --height 224 --width 224 --lr 0.00035 -ds dukemtmc -a ViT --seed 0 --margin 0.0 \
	--num-instances 4 -b 32 -j 4 --warmup-step 10  --iters 520 --milestones 40 70 --epochs 80 \
	--eval-step 5 --logs-dir logs/dukemtmcTOmarket1501/ViT


python3 sbs_traindbscan_unc.py --height 224 --width 224  --lr 0.001  --schedule_step 10 -st dukemtmc -tt market1501 -a ViT\
	--num-instances 4 --iters 400 -b 32 --epochs 200 \
	--eval-step 10 --dropout 0 --n-jobs 16 --choice_c 0 \
	--init-1 logs/dukemtmcTOmarket1501/ViT/model_best.pth.tar \
	--logs-dir logs/dbscan-dukemtmcTOmarket1501/ViT_uncertainly

python3 sbs_traindbscan_unc.py --height 224 --width 224  --lr 0.001  --schedule_step 10 -tt dukemtmc -st market1501 -a ViT\
	--num-instances 4 --iters 400 -b 32 --epochs 200 \
	--eval-step 10 --dropout 0 --n-jobs 16 --choice_c 0 \
	--init-1 logs/market1501TOdukemtmc/ViT/model_best.pth.tar \
	--logs-dir logs/dbscan-market1501TOdukemtmc/ViT_uncertainly