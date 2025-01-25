#!/bin/bash

python3 cross_test.py --name CASwin10_all_trick_duke --which_epoch 10 --test market ; python3 cross_test.py --name CASwin10_all_trick_duke --which_epoch 10 --test duke; python3 cross_test.py --name CASwin10_all_trick_duke --which_epoch 10 --test cuhk03

python3 cross_test.py --name CASwin20_all_trick_duke --which_epoch 20 --test market ; python3 cross_test.py --name CASwin20_all_trick_duke --which_epoch 20 --test duke; python3 cross_test.py --name CASwin20_all_trick_duke --which_epoch 20 --test cuhk03

python3 cross_test.py --name CASwin30_all_trick_duke --which_epoch 30 --test market ; python3 cross_test.py --name CASwin30_all_trick_duke --which_epoch 30 --test duke; python3 cross_test.py --name CASwin30_all_trick_duke --which_epoch 30 --test cuhk03

python3 cross_test.py --name CASwin40_all_trick_duke --which_epoch 40 --test market ; python3 cross_test.py --name CASwin40_all_trick_duke --which_epoch 40 --test duke; python3 cross_test.py --name CASwin40_all_trick_duke --which_epoch 40 --test cuhk03

python3 cross_test.py --name CASwin50_all_trick_duke --which_epoch 50 --test market ; python3 cross_test.py --name CASwin50_all_trick_duke --which_epoch 50 --test duke; python3 cross_test.py --name CASwin50_all_trick_duke --which_epoch 50 --test cuhk03
