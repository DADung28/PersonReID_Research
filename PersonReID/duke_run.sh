#!/bin/bash 
python3 train.py --use_dense --data_dir /home/jun/ReID_Dataset/duke/dataloader/ --lr 0.01 --batch 32 --triplet --warm_epoch 5 --name DenseNet_all_trick_duke --gpu_ids $1
python3 cross_test.py --name DenseNet_all_trick_duke --test duke --gpu_ids $1
python3 cross_test.py --name DenseNet_all_trick_duke --test market --gpu_ids $1
python3 cross_test.py --name DenseNet_all_trick_duke --test cuhk03 --gpu_ids $1
#!/bin/bash 
python3 train.py --use_efficient --data_dir /home/jun/ReID_Dataset/duke/dataloader/ --lr 0.01 --batch 32 --triplet --warm_epoch 5 --name EfficientNet_all_trick_duke --gpu_ids $1
python3 cross_test.py --name EfficientNet_all_trick_duke --test duke --gpu_ids $1
python3 cross_test.py --name EfficientNet_all_trick_duke --test market --gpu_ids $1
python3 cross_test.py --name EfficientNet_all_trick_duke --test cuhk03 --gpu_ids $1
#!/bin/bash 
python3 train.py --use_lenet --data_dir /home/jun/ReID_Dataset/duke/dataloader/ --lr 0.01 --batch 32 --triplet --warm_epoch 5 --name GoogLeNet_all_trick_duke --gpu_ids $1
python3 cross_test.py --name GoogLeNet_all_trick_duke --test duke --gpu_ids $1
python3 cross_test.py --name GoogLeNet_all_trick_duke --test market --gpu_ids $1
python3 cross_test.py --name GoogLeNet_all_trick_duke --test cuhk03 --gpu_ids $1
#!/bin/bash 
python3 train.py --use_hr --data_dir /home/jun/ReID_Dataset/duke/dataloader/ --lr 0.01 --batch 32 --triplet --warm_epoch 5 --name HRNet_all_trick_duke --gpu_ids $1
python3 cross_test.py --name HRNet_all_trick_duke --test duke --gpu_ids $1
python3 cross_test.py --name HRNet_all_trick_duke --test market --gpu_ids $1
python3 cross_test.py --name HRNet_all_trick_duke --test cuhk03 --gpu_ids $1
#!/bin/bash 
python3 train.py --data_dir /home/jun/ReID_Dataset/duke/dataloader/ --lr 0.01 --batch 32 --triplet --warm_epoch 5 --name ResNet50_all_trick_duke --gpu_ids $1
python3 cross_test.py --name ResNet50_all_trick_duke --test duke --gpu_ids $1
python3 cross_test.py --name ResNet50_all_trick_duke --test market --gpu_ids $1
python3 cross_test.py --name ResNet50_all_trick_duke --test cuhk03 --gpu_ids $1
#!/bin/bash 
python3 train.py --use_swin --data_dir /home/jun/ReID_Dataset/duke/dataloader/ --lr 0.01 --batch 32 --triplet  --warm_epoch 5 --name Swin_all_trick_duke --gpu_ids $1
python3 cross_test.py --name Swin_all_trick_duke --test duke --gpu_ids $1
python3 cross_test.py --name Swin_all_trick_duke --test market --gpu_ids $1
python3 cross_test.py --name Swin_all_trick_duke --test cuhk03 --gpu_ids $1
#!/bin/bash 
python3 train.py --use_vit --data_dir /home/jun/ReID_Dataset/duke/dataloader/ --lr 0.01 --batch 32 --triplet --warm_epoch 5 --name ViT_all_trick_duke --gpu_ids $1
python3 cross_test.py --name ViT_all_trick_duke --test duke --gpu_ids $1
python3 cross_test.py --name ViT_all_trick_duke --test market --gpu_ids $1
python3 cross_test.py --name ViT_all_trick_duke --test cuhk03 --gpu_ids $1
