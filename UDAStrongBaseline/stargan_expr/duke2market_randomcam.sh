#!/bin/sh
CUDA_VISIBLE_DEVICES=0 \

#python3 source_pretrain.py -train dukemtmc_stargan_randomcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
#   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-stargan-singlecam\
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
#	--eval-step 10 --logs-dir logs/stargan_expr/duke2market/ResNet50_stargan_singlecam_randomcam
#
#python3 source_pretrain.py -train dukemtmc_stargan_randomcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
#   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-stargan-0_0\
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
#	--eval-step 10 --logs-dir logs/stargan_expr/duke2market/ResNet50_stargan_0_0_randomcam
#
#python3 source_pretrain.py -train dukemtmc_stargan_randomcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
#   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-stargan-0_1 \
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
#	--eval-step 10 --logs-dir logs/stargan_expr/duke2market/ResNet50_stargan_0_1_randomcam
#
#python3 source_pretrain.py -train dukemtmc_stargan_randomcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
#   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-stargan-0_2 \
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
#	--eval-step 10 --logs-dir logs/stargan_expr/duke2market/ResNet50_stargan_0_2_randomcam
#
#python3 source_pretrain.py -train dukemtmc_stargan_randomcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
#   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-stargan-0_5 \
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
#	--eval-step 10 --logs-dir logs/stargan_expr/duke2market/ResNet50_stargan_0_5_randomcam
#
#
#python3 source_pretrain.py -train dukemtmc_stargan_randomcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
#   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-stargan-1_0 \
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
#	--eval-step 10 --logs-dir logs/stargan_expr/duke2market/ResNet50_stargan_1_0_randomcam
#
#python3 source_pretrain.py -train dukemtmc_stargan_randomcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
#   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-stargan-1_1 \
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
#	--eval-step 10 --logs-dir logs/stargan_expr/duke2market/ResNet50_stargan_1_1_randomcam
#
#python3 source_pretrain.py -train dukemtmc_stargan_randomcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
#   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-stargan-1_2 \
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
#	--eval-step 10 --logs-dir logs/stargan_expr/duke2market/ResNet50_stargan_1_2_randomcam
#
python3 source_pretrain.py -train dukemtmc_stargan_randomcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-stargan-1_5 \
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/stargan_expr/duke2market/ResNet50_stargan_1_5_randomcam

python3 source_pretrain.py -train dukemtmc_stargan_randomcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-stargan-2_0\
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/stargan_expr/duke2market/ResNet50_stargan_2_0_randomcam

python3 source_pretrain.py -train dukemtmc_stargan_randomcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-stargan-2_1\
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/stargan_expr/duke2market/ResNet50_stargan_2_1_randomcam

python3 source_pretrain.py -train dukemtmc_stargan_randomcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-stargan-2_2\
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/stargan_expr/duke2market/ResNet50_stargan_2_2_randomcam

python3 source_pretrain.py -train dukemtmc_stargan_randomcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-stargan-2_5\
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/stargan_expr/duke2market/ResNet50_stargan_2_5_randomcam

python3 source_pretrain.py -train dukemtmc_stargan_randomcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-stargan-5_0\
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/stargan_expr/duke2market/ResNet50_stargan_5_0_randomcam

python3 source_pretrain.py -train dukemtmc_stargan_randomcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-stargan-5_1\
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/stargan_expr/duke2market/ResNet50_stargan_5_1_randomcam

python3 source_pretrain.py -train dukemtmc_stargan_randomcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-stargan-5_2\
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/stargan_expr/duke2market/ResNet50_stargan_5_2_randomcam

python3 source_pretrain.py -train dukemtmc_stargan_randomcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --data-dir /home/jun/ReID_Dataset/DukeMTMC-reID-stargan-5_5\
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --epochs 200 \
	--eval-step 10 --logs-dir logs/stargan_expr/duke2market/ResNet50_stargan_5_5_randomcam
