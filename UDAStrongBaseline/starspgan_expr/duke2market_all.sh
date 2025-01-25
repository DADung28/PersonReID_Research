#!/bin/sh
CUDA_VISIBLE_DEVICES=0 \

# Single camera
python3 source_pretrain.py -train dukemtmc_stargan_allcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-starspgan-singlecam_0_0\
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/starspgan_expr/duke2market/ResNet50_starspgan_singlecam_0_0_all

python3 source_pretrain.py -train dukemtmc_stargan_allcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-starspgan-singlecam_0_5\
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/starspgan_expr/duke2market/ResNet50_starspgan_singlecam_0_5_all

python3 source_pretrain.py -train dukemtmc_stargan_allcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-starspgan-singlecam_1_0\
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/starspgan_expr/duke2market/ResNet50_starspgan_singlecam_1_0_all

python3 source_pretrain.py -train dukemtmc_stargan_allcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-starspgan-singlecam_1_5\
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/starspgan_expr/duke2market/ResNet50_starspgan_singlecam_1_5_all

python3 source_pretrain.py -train dukemtmc_stargan_allcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-starspgan-singlecam_2_0\
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/starspgan_expr/duke2market/ResNet50_starspgan_singlecam_2_0_all

python3 source_pretrain.py -train dukemtmc_stargan_allcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-starspgan-singlecam_2_5\
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/starspgan_expr/duke2market/ResNet50_starspgan_singlecam_2_5_all





# Multi camera
python3 source_pretrain.py -train dukemtmc_stargan_allcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-starspgan-0_0\
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/starspgan_expr/duke2market/ResNet50_starspgan_0_0_all

python3 source_pretrain.py -train dukemtmc_stargan_allcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-starspgan-0_5\
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/starspgan_expr/duke2market/ResNet50_starspgan_0_5_all

python3 source_pretrain.py -train dukemtmc_stargan_allcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-starspgan-1_0\
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/starspgan_expr/duke2market/ResNet50_starspgan_1_0_all

python3 source_pretrain.py -train dukemtmc_stargan_allcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-starspgan-1_5\
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/starspgan_expr/duke2market/ResNet50_starspgan_1_5_all

python3 source_pretrain.py -train dukemtmc_stargan_allcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-starspgan-2_0\
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/starspgan_expr/duke2market/ResNet50_starspgan_2_0_all

python3 source_pretrain.py -train dukemtmc_stargan_allcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-starspgan-2_5\
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/starspgan_expr/duke2market/ResNet50_starspgan_2_5_all