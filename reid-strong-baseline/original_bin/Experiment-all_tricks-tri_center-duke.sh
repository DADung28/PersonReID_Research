# Experiment all tricks with center loss : 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005
# Dataset 2: dukemtmc
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# with center loss
python3 tools/train.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('dukemtmc')" DATASETS.ROOT_DIR "('/home/jun/ReID_Dataset')" OUTPUT_DIR "('/home/jun/reid_strong_baseline/log/Original')"