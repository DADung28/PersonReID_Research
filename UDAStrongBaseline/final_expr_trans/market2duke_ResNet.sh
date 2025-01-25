#!/bin/sh
CUDA_VISIBLE_DEVICES=0 \

# DBSCAN Baseline
python3 sbs_traindbscan.py --height 256 --width 128  --lr 0.00035 -tt dukemtmc -st market1501 -a ResNet50 --train_type target \
	--num-instances 16 --iters 200 -b 64 --epochs 200 --eval-step 4\
	--dropout 0 --n-jobs 16 --choice_c 0 \
	--init-1 /home/jun/UDAStrongBaseline/logs/final_expr/market2duke/ResNet50_0/model_best.pth.tar \
	--logs-dir /home/jun/UDAStrongBaseline/logs/final_expr/market2duke/ResNet50_0/UDA_dbscan

python3 sbs_traindbscan.py --height 256 --width 128  --lr 0.00035 -tt dukemtmc -st market1501 -a ResNet50 --train_type target \
	--num-instances 16 --iters 200 -b 64 --epochs 200 --eval-step 4\
	--dropout 0 --n-jobs 16 --choice_c 0 \
	--init-1 /home/jun/UDAStrongBaseline/logs/final_expr/market2duke/ResNet50_spgan_randomcam_0/model_best.pth.tar \
	--logs-dir /home/jun/UDAStrongBaseline/logs/final_expr/market2duke/ResNet50_spgan_randomcam_0/UDA_dbscan

# DBSCAN uncertantly
python3 sbs_traindbscan_unc.py --height 256 --width 128 --lr 0.00035  -tt dukemtmc -st market1501 -a ResNet50\
	--num-instances 16 --iters 200 -b 64 --epochs 200 \
	--dropout 0 --n-jobs 16 --choice_c 0 \
	--init-1 /home/jun/UDAStrongBaseline/logs/final_expr/duke2market/ResNet50_0/model_best.pth.tar \
	--logs-dir /home/jun/UDAStrongBaseline/logs/final_expr/market2duke/ResNet50_0/UDA_dbscan_unc

# DBSCAN uncertantly
python3 sbs_traindbscan_unc.py --height 256 --width 128 --lr 0.00035  -tt dukemtmc -st market1501 -a ResNet50\
	--num-instances 16 --iters 200 -b 64 --epochs 200 \
	--dropout 0 --n-jobs 16 --choice_c 0 \
	--init-1 /home/jun/UDAStrongBaseline/logs/final_expr/duke2market/ResNet50_spgan_randomcam_0/model_best.pth.tar \
	--logs-dir /home/jun/UDAStrongBaseline/logs/final_expr/market2duke/ResNet50_spgan_randomcam_0/UDA_dbscan_unc

