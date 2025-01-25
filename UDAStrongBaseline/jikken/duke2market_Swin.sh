#!/bin/sh
CUDA_VISIBLE_DEVICES=0 \

#python3 source_pretrain.py -train dukemtmc -dt market1501 --height 224 --width 224 -ds dukemtmc -a Swin --seed 0 --margin 0.0 \
#	--num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.001 --milestones 40 70 --iters 200 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/duke2market/Swin_0.001

#python3 source_pretrain.py -train dukemtmc -dt market1501 --height 224 --width 224 -ds dukemtmc -a Swin --seed 0 --margin 0.0 \
#	--num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/duke2market/Swin_0.00035

#python3 source_pretrain.py -train dukemtmc_all -dt market1501 --height 224 --width 224 -ds dukemtmc -a Swin --seed 0 --margin 0.0 \
#	--num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.001 --milestones 40 70 --iters 400 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/duke2market/Swin_all_400_0.001

#python3 source_pretrain.py -train dukemtmc_all -dt market1501 --height 224 --width 224 -ds dukemtmc -a Swin --seed 0 --margin 0.0 \
#	--num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 400 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/duke2market/Swin_all_400_0.00035

#python3 source_pretrain.py -train dukemtmc_all -dt market1501 --height 224 --width 224 -ds dukemtmc -a Swin --seed 0 --margin 0.0 \
#	--num-instances 16 -b 64 -j 4 --warmup-step 10 --lr 0.001 --milestones 40 70 --iters 200 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/duke2market/Swin_all_200_0.001

#python3 source_pretrain.py -train dukemtmc_all -dt market1501 --height 224 --width 224 -ds dukemtmc -a Swin --seed 0 --margin 0.0 \
#	--num-instances 16 -b 63 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/duke2market/Swin_all_200_0.00035

python3 source_pretrain.py -train dukemtmc_marketstyle -dt market1501 --height 224 --width 224 -ds dukemtmc -a Swin --seed 0 --margin 0.0 \
	--num-instances 16 -b 63 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 100 \
	--eval-step 10 --logs-dir logs/jikken/duke2market/Swin_SPGAN_200_0.00035




#python3 source_pretrain.py -dt market1501 --height 224 --width 224 -ds dukemtmc_marketstyle -a Swin --seed 0 --margin 0.0 \
#	--num-instances 4 -b 32 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 400 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/duke2market/Swin_gan_0.00035

#python3 source_pretrain.py -dt market1501 --height 224 --width 224 -ds dukemtmc_all -a Swin --seed 0 --margin 0.0 \
#	--num-instances 4 -b 32 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 400 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/duke2market/Swin_all_0.00035

#python3 source_pretrain.py -dt market1501 --height 224 --width 224 -ds dukemtmc_marketstyle -a Swin --seed 0 --margin 0.0 \
#	--num-instances 4 -b 32 -j 4 --warmup-step 10 --lr 0.001 --milestones 40 70 --iters 400 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/duke2market/Swin_gan_0.001

#python3 source_pretrain.py -dt market1501 --height 224 --width 224 -ds dukemtmc -a Swin --seed 0 --margin 0.0 \
#	--num-instances 4 -b 32 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 400 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/duke2market/Swin

#python3 source_pretrain.py -dt market1501 --height 224 --width 224 -ds dukemtmc -a Swin --seed 0 --margin 0.0 \
#	--num-instances 4 -b 32 -j 4 --warmup-step 10 --lr 0.001 --milestones 40 70 --iters 400 --epochs 100 \
#	--eval-step 10 --logs-dir logs/jikken/duke2market/Swin_0.001