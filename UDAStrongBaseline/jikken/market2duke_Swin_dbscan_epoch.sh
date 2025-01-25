#!/bin/sh
CUDA_VISIBLE_DEVICES=0 \
#python3 source_pretrain.py -ds market1501 --height 224 --width 224 -dt dukemtmc -a Swin --seed 0 --margin 0.0 \
#	--num-instances 4 -b 32 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 400 --epochs 100 \
#	--eval-step 10 --logs-dir logs/market1501TOdukemtmc/Swin_0.00035

# DBSCAN Baseline

python3 sbs_traindbscan.py -tt dukemtmc --height 224 --width 224  --lr 0.001 -st market1501 -a Swin --train_type target\
	--num-instances 4 --iters 400 -b 32 --epochs 200 --eval-step 5 --dbscan_epoch 10\
	--dropout 0 --n-jobs 16 --choice_c 0 \
	--init-1 logs/market1501TOdukemtmc/Swin/model_best.pth.tar \
	--logs-dir logs/dbscan-market1501TOdukemtmc/Swin_tar_0.001_ce_tri_10step

python3 sbs_traindbscan.py -tt dukemtmc --height 224 --width 224  --lr 0.001 -st market1501 -a Swin --train_type source+target\
	--num-instances 4 --iters 400 -b 32 --epochs 200 --eval-step 5 --dbscan_epoch 10\
	--dropout 0 --n-jobs 16 --choice_c 0 \
	--init-1 logs/market1501TOdukemtmc/Swin/model_best.pth.tar \
	--logs-dir logs/dbscan-market1501TOdukemtmc/Swin_both_0.001_ce_tri_10step

python3 sbs_traindbscan.py -tt dukemtmc --height 224 --width 224  --lr 0.0005 -st market1501 -a Swin --train_type target\
	--num-instances 4 --iters 400 -b 32 --epochs 200 --eval-step 5 --dbscan_epoch 10\
	--dropout 0 --n-jobs 16 --choice_c 0 \
	--init-1 logs/market1501TOdukemtmc/Swin/model_best.pth.tar \
	--logs-dir logs/dbscan-market1501TOdukemtmc/Swin_tar_0.0005_ce_tri_10step

python3 sbs_traindbscan.py -tt dukemtmc --height 224 --width 224  --lr 0.0005 -st market1501 -a Swin --train_type source+target\
	--num-instances 4 --iters 400 -b 32 --epochs 200 --eval-step 5 --dbscan_epoch 10\
	--dropout 0 --n-jobs 16 --choice_c 0 \
	--init-1 logs/market1501TOdukemtmc/Swin/model_best.pth.tar \
	--logs-dir logs/dbscan-market1501TOdukemtmc/Swin_both_0.0005_ce_tri_10step

# DBSCAN uncertantly
#python3 sbs_traindbscan_unc.py --height 224 --width 224  --lr 0.00035  --schedule_step 10 -tt dukemtmc -st market1501 -a Swin\
#	--num-instances 4 --iters 400 -b 32 --epochs 200 \
#	--dropout 0 --n-jobs 16 --choice_c 0 \
#	--init-1 logs/market1501TOdukemtmc/Swin_0.00035/model_best.pth.tar \
#	--logs-dir logs/dbscan-market1501TOdukemtmc/Swin_uncertainly_0.00035

