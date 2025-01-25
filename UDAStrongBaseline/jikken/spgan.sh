#!/bin/sh
CUDA_VISIBLE_DEVICES=0 \

#python3 source_pretrain.py -ds market1501_dukestyle --height 224 --width 224 -dt dukemtmc -a Swin --seed 0 --margin 0.0 \
#	--num-instances 8 -b 32 -j 4 --warmup-step 10 --lr 0.001 --milestones 40 70 --iters 400 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/market2duke/Swin_gan_0.001

#python3 source_pretrain.py -ds market1501_all --height 224 --width 224 -dt dukemtmc -a Swin --seed 0 --margin 0.0 \
#	--num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.001 --milestones 40 70 --iters 400 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/market2duke/Swin_all_400_0.001

#python3 source_pretrain.py -ds market1501_all --height 224 --width 224 -dt dukemtmc -a Swin --seed 0 --margin 0.0 \
#	--num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 400 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/market2duke/Swin_all_400_0.00035



#python3 source_pretrain.py -ds market1501_all --height 224 --width 224 -dt dukemtmc -a Swin --seed 0 --margin 0.0 \
#	--num-instances 8 -b 32 -j 4 --warmup-step 10 --lr 0.001 --milestones 40 70 --iters 800 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/market2duke/Swin_all_800_0.001

#python3 source_pretrain.py -ds market1501_all --height 224 --width 224 -dt dukemtmc -a Swin --seed 0 --margin 0.0 \
#	--num-instances 8 -b 32 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 800 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/market2duke/Swin_all_800_0.00035

#python3 source_pretrain.py -ds market1501_dukestyle --height 224 --width 224 -dt dukemtmc -a Swin --seed 0 --margin 0.0 \
#	--num-instances 8 -b 32 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 400 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/market2duke/Swin_gan_0.00035

#python3 source_pretrain.py -ds market1501 --height 224 --width 224 -dt dukemtmc -a Swin --seed 0 --margin 0.0 \
#	--num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/market2duke/Swin_0.00035


#python3 source_pretrain.py -train market1501 -ds market1501 --height 224 --width 224 -dt dukemtmc -a Swin --seed 0 --margin 0.0 \
#	--num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.001 --milestones 40 70 --iters 200 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/market2duke/Swin_0.001


#python3 source_pretrain.py -train market1501_all -ds market1501  --height 224 --width 224 -dt dukemtmc -a Swin --seed 0 --margin 0.0 \
#	--num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.001 --milestones 40 70 --iters 200 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/market2duke/Swin_all_200_0.001

#python3 source_pretrain.py -train market1501_all -ds market1501 --height 224 --width 224 -dt dukemtmc -a Swin --seed 0 --margin 0.0 \
#	--num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.0005 --milestones 40 70 --iters 200 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/market2duke/Swin_all_200_0.0005_cycle_all

#python3 source_pretrain.py -train market1501_cycleGAN -ds market1501 --height 224 --width 224 -dt dukemtmc -a Swin --seed 0 --margin 0.0 \
#	--num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.001 --milestones 40 70 --iters 200 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/market2duke/Swin_all_200_0.001_cycle

#python3 source_pretrain.py -train market1501_dukestyle -ds market1501 --height 224 --width 224 -dt dukemtmc -a Swin --seed 0 --margin 0.0 \
#	--num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.001 --milestones 40 70 --iters 200 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/market2duke/Swin_dukestyle_200_0.001

#python3 source_pretrain.py -train market1501_dukestyle -ds market1501 --height 224 --width 224 -dt dukemtmc -a Swin --seed 0 --margin 0.0 \
#	--num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/market2duke/Swin_dukestyle_200_0.00035


#python3 source_pretrain.py -train market1501_random -ds market1501 --height 256 --width 128 -dt dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
#	--num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/market2duke/ResNet_spgan_random_200_0.00035

#python3 source_pretrain.py -train market1501_all -ds market1501  --height 256 --width 128 -dt dukemtmc -a ResNet50 --adam --seed 0 --margin 0.0 \
#	--num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/market2duke/ResNet_spgan_all_200_0.00035

#python3 source_pretrain.py -train dukemtmc_random -ds dukemtmc --height 256 --width 128 -dt market1501 -a ResNet50 --adam --seed 0 --margin 0.0 \
#	--num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/duke2market/ResNet_spgan_random_200_0.00035

#python3 source_pretrain.py -train dukemtmc_all -ds dukemtmc  --height 256 --width 128 -dt market1501 -a ResNet50 --adam --seed 0 --margin 0.0 \
#	--num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/duke2market/ResNet_spgan_all_200_0.00035


python3 source_pretrain.py -train market1501_random -ds market1501 --height 224 --width 224 -dt dukemtmc -a Swin --seed 0 --margin 0.0 \
	--num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 100 \
	--eval-step 10 --logs-dir logs/jikken/market2duke/Swin_spgan_random_200_0.00035

python3 source_pretrain.py -train market1501_all -ds market1501  --height 224 --width 224 -dt dukemtmc -a Swin  --seed 0 --margin 0.0 \
	--num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 100 \
	--eval-step 10 --logs-dir logs/jikken/market2duke/Swin_spgan_all_200_0.00035

python3 source_pretrain.py -train dukemtmc_random -ds dukemtmc --height 224 --width 224 -dt market1501 -a Swin  --seed 0 --margin 0.0 \
	--num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 100 \
	--eval-step 10 --logs-dir logs/jikken/duke2market/Swin_spgan_random_200_0.00035

python3 source_pretrain.py -train dukemtmc_all -ds dukemtmc  --height 224 --width 224 -dt market1501 -a Swin  --seed 0 --margin 0.0 \
	--num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 100 \
	--eval-step 10 --logs-dir logs/jikken/duke2market/Swin_spgan_all_200_0.00035