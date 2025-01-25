#!/bin/sh
CUDA_VISIBLE_DEVICES=0 \

#python3 source_pretrain.py -train dukemtmc_spgan_allcam -ds market1501  --height 256 --width 128 -dt dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 1400 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/duke2market/ResNet_spgan_allcam_1400_0.00035


#python3 source_pretrain.py -train dukemtmc_spgan_allcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
#  --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 100 \
#  --eval-step 10 --logs-dir logs/jikken/duke2market/ResNet_spgan_allcam_200_0.00035


python3 source_pretrain.py -train dukemtmc_spgan_randomcam -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 100 \
	--eval-step 10 --logs-dir logs/jikken/duke2market/ResNet_spgan_randomcam_new_200_0.00035

#python3 source_pretrain.py -train dukemtmc_spgan_random -dt market1501  --height 256 --width 128 -ds dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
#  --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 100 \
#  --eval-step 10 --logs-dir logs/jikken/duke2market/ResNet_spgan_random_200_0.00035

#python3 source_pretrain.py -train dukemtmc_spgan_randomcam -dt market1501  --height 224 --width 224 -ds dukemtmc -a Swin  --seed 0 --margin 0.0 \
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/duke2market/Swin_spgan_randomcam_200_0.00035

#python3 source_pretrain.py -train dukemtmc_spgan_randomcam -dt market1501  --height 224 --width 224 -ds dukemtmc -a Swin  --seed 0 --margin 0.0 \
#   --num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.001 --milestones 40 70 --iters 200 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/duke2market/Swin_spgan_randomcam_200_0.001