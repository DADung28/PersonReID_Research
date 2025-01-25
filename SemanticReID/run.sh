#!/bin/bash

python3 train.py --TwinViT --name TwinViT_v2 --circle ; python3 cross_test.py --name TwinViT_v2 --test duke  
python3 train.py --TwinSwin --name TwinSwin --circle ; python3 cross_test.py --name TwinSwin --test duke  
