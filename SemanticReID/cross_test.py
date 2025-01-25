# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from tqdm import tqdm
from model import * 
from utils.utils import fuse_all_conv_bn

######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Test cross dataset')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--data_dir',default='/home/jun/ReID_Dataset/market/dataloader',type=str, help='./test_data')
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--name', default='ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=16, type=int, help='batchsize')
parser.add_argument('--linear_num', default=1024, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
parser.add_argument('--test',default='duke', type=str,help='Choose test dataset type for test')
opt = parser.parse_args()
###load config###
# load the training config
config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader) # for the new pyyaml via 'conda install pyyaml'
opt.arcface = config['arcface']
opt.circle = config['circle']
opt.cosface = config['cosface']
opt.contrast = config['contrast']
opt.instance = config['instance']
opt.triplet = config['triplet']
opt.lifted = config['lifted']
opt.sphere = config['sphere']
if 'vit' in config:      
    opt.vit = config['vit']        
else:
    opt.vit = False
    
if 'swin' in config:      
    opt.swin = config['swin']        
else:
    opt.swin = False
    
if 'pivit' in config:      
    opt.pivit = config['pivit']        
else:
    opt.pivit = False    
    
if 'pivit_nohead' in config:      
    opt.pivit_nohead = config['pivit_nohead']        
else:
    opt.pivit_nohead = False 
    
if 'pivit_4part' in config:      
    opt.pivit_4part = config['pivit_4part']        
else:
    opt.pivit_4part = False   

if 'pivit_3part' in config:      
    opt.pivit_3part = config['pivit_3part']        
else:
    opt.pivit_3part = False     
    
if 'centroid' in config:      
    opt.centroid = config['centroid']        
else:
    opt.centroid = False
    
if 'semanticRes' in config:      
    opt.semanticRes = config['semanticRes']        
else:
    opt.semanticRes = False
    
if 'semanticVit' in config:      
    opt.semanticVit = config['semanticVit']        
else:
    opt.semanticVit = False

if 'TwinViT' in config:      
    opt.TwinViT = config['TwinViT']        
else:
    opt.TwinViT = False
    
if 'TwinSwin' in config:      
    opt.TwinSwin = config['TwinSwin']        
else:
    opt.TwinSwin = False
    
if 'SPResNet50' in config:      
    opt.SPResNet50 = config['SPResNet50']        
else:
    opt.SPResNet50 = False
    
if 'resnet_4head' in config:      
    opt.resnet_4head = config['resnet_4head']        
else:
    opt.resnet_4head = False
    
if 'piresnet' in config:      
    opt.piresnet = config['piresnet']        
else:
    opt.piresnet = False    

if 'piswin' in config:      
    opt.piswin = config['piswin']        
else:
    opt.piswin = False   

if 'TwinResNet50' in config:      
    opt.TwinResNet50 = config['TwinResNet50']        
else:
    opt.TwinResNet50 = False  
    
if 'h' in config:      
    opt.h = config['h']        
else:
    opt.h = False

if 'w' in config:      
    opt.w = config['w']        
else:
    opt.w = False
    
opt.stride = config['stride']

opt.data_dir = config['data_dir']
if opt.data_dir.find('market1501') != -1:
    opt.data_dir = opt.data_dir.replace('market1501','market')
elif opt.data_dir.find('dukemtmcreid') != -1:
    opt.data_dir = opt.data_dir.replace('dukemtmcreid','duke')
else:
    pass

print(f'Reading data from {opt.data_dir}')
train_type = opt.data_dir.split('/')[4]
if opt.test == 'market':
    opt.data_dir = '/home/jun/ReID_Dataset/market/dataloader'
elif opt.test == 'duke':
    opt.data_dir = '/home/jun/ReID_Dataset/duke/dataloader'
elif opt.test == 'cuhk03':
    opt.data_dir = '/home/jun/ReID_Dataset/cuhk03/dataloader_new_detected'

elif opt.test == 'market_TwinPic_grayscale_background':
    opt.data_dir = '/home/jun/ReID_Dataset/market_TwinPic_grayscale_background/dataloader'
elif opt.test == 'duke_TwinPic_grayscale_background':
    opt.data_dir = '/home/jun/ReID_Dataset/duke_TwinPic_grayscale_background/dataloader'
elif opt.test == 'cuhk03_TwinPic_grayscale_background':
    opt.data_dir = '/home/jun/ReID_Dataset/cuhk03_TwinPic_grayscale_background/dataloader_new_detected'

elif opt.test == 'market_TwinPic_delete_background':
    opt.data_dir = '/home/jun/ReID_Dataset/market_TwinPic_delete_background/dataloader'
elif opt.test == 'duke_TwinPic_delete_background':
    opt.data_dir = '/home/jun/ReID_Dataset/duke_TwinPic_delete_background/dataloader'
elif opt.test == 'cuhk03_TwinPic_delete_background':
    opt.data_dir = '/home/jun/ReID_Dataset/cuhk03_TwinPic_delete_background/dataloader_new_detected'

elif opt.test == 'market_SinglePic_delete_background':
    opt.data_dir = '/home/jun/ReID_Dataset/market_SinglePic_delete_background/dataloader'
elif opt.test == 'duke_SinglePic_delete_background':
    opt.data_dir = '/home/jun/ReID_Dataset/duke_SinglePic_delete_background/dataloader'
elif opt.test == 'cuhk03_SinglePic_delete_background':
    opt.data_dir = '/home/jun/ReID_Dataset/cuhk03_SinglePic_delete_background/dataloader_new_detected'

elif opt.test == 'market_SinglePic_grayscale_background':
    opt.data_dir = '/home/jun/ReID_Dataset/market_SinglePic_grayscale_background/dataloader'
elif opt.test == 'duke_SinglePic_grayscale_background':
    opt.data_dir = '/home/jun/ReID_Dataset/duke_SinglePic_grayscale_background/dataloader'
elif opt.test == 'cuhk03_SinglePic_grayscale_background':
    opt.data_dir = '/home/jun/ReID_Dataset/cuhk03_SinglePic_grayscale_background/dataloader_new_detected'

elif opt.test == 'market_PI':
    opt.data_dir = '/home/jun/ReID_Dataset/market_PI/dataloader'
elif opt.test == 'duke_PI':
    opt.data_dir = '/home/jun/ReID_Dataset/duke_PI/dataloader'
elif opt.test == 'cuhk03_PI':
    opt.data_dir = '/home/jun/ReID_Dataset/cuhk03_PI/dataloader_new_detected'
else:
    pass
if opt.data_dir.find('cuhk03') != -1:
    cuhk03_dataset = True
else:
    cuhk03_dataset = False 
test_type = opt.data_dir.split('/')[4]
print(f'Save file in result_{train_type}_{test_type}.txt')
if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 751 
if 'ibn' in config:
    opt.ibn = config['ibn']
if 'linear_num' in config:
    opt.linear_num = config['linear_num']
str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
data_dir = opt.data_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))
print(gpu_ids) 
# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#

h, w = opt.h, opt.w

data_transforms = transforms.Compose([
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                            shuffle=False, num_workers=16) for x in ['gallery','query']}
class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    print(f'Loading weight from {save_path}')
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    #features = torch.FloatTensor()
    # count = 0
    pbar = tqdm()

    for iter, data in enumerate(dataloaders):
        img, label = data
        n, c, h, w = img.size()
        # count += n
        # print(count)
        pbar.update(n)
        if opt.semanticRes:
            ff = torch.FloatTensor(n,opt.linear_num).zero_().cuda() 
        elif opt.semanticVit or opt.TwinViT or opt.TwinSwin or opt.TwinResNet50:
            ff = torch.FloatTensor(n,opt.linear_num*2).zero_().cuda()
        elif opt.resnet_4head:
            ff = torch.FloatTensor(n,opt.linear_num*4).zero_().cuda()  
        else:
            ff = torch.FloatTensor(n,opt.linear_num).zero_().cuda()
        
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if not torch.cuda.is_available():
                    input_img = nn.functional.interpolate(input_img.cpu(), scale_factor=scale, mode='bicubic', align_corners=False)
                else:
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img)
                #print(outputs.shape, ff.shape)
                
                ff += outputs
                
                #ff = torch.cat((ff, outputs), 0)
        
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        if iter == 0:
            features = torch.FloatTensor( len(dataloaders.dataset), ff.shape[1])
            
        start = iter*opt.batchsize
        end = min( (iter+1)*opt.batchsize, len(dataloaders.dataset))
        features[start:end,:] = ff
        #print(f'ff shape: {ff.shape}')
        #print(f'features shape: {features.shape}, from {start} to {end}')
    pbar.close()
    return features

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

if cuhk03_dataset:
    def get_id(dataset):
        camera_id = []
        labels = []
        for i, (img, label) in enumerate(dataset):
            #filename = path.split('/')[-1]
            filename = os.path.basename(gallery_path[i][0])
            camera = filename.split('_')[2]
            #print(f'Pic path: {filename}, Pic label: {label}, Pic camera: {camera}')
            labels.append(int(label))
            camera_id.append(int(camera))
        return camera_id, labels
    
    gallery_cam,gallery_label = get_id(image_datasets['gallery'])
    query_cam,query_label = get_id(image_datasets['query'])
else:
    def get_id(img_path):
        camera_id = []
        labels = []
        for path, v in img_path:
            #filename = path.split('/')[-1]
            filename = os.path.basename(path)
            label = filename[0:4]
            camera = filename.split('c')[1]
            if label[0:2]=='-1':
                labels.append(-1)
            else:
                labels.append(int(label))
            camera_id.append(int(camera[0]))
        return camera_id, labels
    
    gallery_cam,gallery_label = get_id(gallery_path)
    query_cam,query_label = get_id(query_path)


######################################################################
# Load Collected data Trained model
print('-------test-----------')
return_feature = opt.arcface or opt.cosface or opt.circle or opt.triplet or opt.contrast or opt.instance or opt.lifted or opt.sphere or opt.centroid
if opt.semanticRes:
    model_structure = ResParsing(opt.nclasses, stride = opt.stride, test = True, linear_num=opt.linear_num)
elif opt.vit:
    model_structure = ViT(opt.nclasses, stride = opt.stride, test = True, linear_num=opt.linear_num, size=h)
elif opt.swin:
    model_structure = Swin(opt.nclasses, test = True, linear_num=opt.linear_num, size=h)
elif opt.TwinViT:
    model_structure = TwinViT(opt.nclasses, test = True, linear_num=opt.linear_num, size=h)
elif opt.pivit:
    model_structure = PIViT(opt.nclasses, test = True, linear_num=opt.linear_num, size=h)
elif opt.piswin:
    model_structure = PISwin(opt.nclasses, test = True, linear_num=opt.linear_num, size=h)
elif opt.piresnet:
    model_structure = PIResNet50(opt.nclasses, test = True, linear_num=opt.linear_num)
elif opt.pivit_4part:
    model_structure = PIViT_4part(opt.nclasses, test = True, linear_num=opt.linear_num, size=h)
elif opt.pivit_nohead:
    model_structure = PIViT_nohead(opt.nclasses, test = True, linear_num=opt.linear_num, size=h)
elif opt.pivit_3part:
    model_structure = PIViT_3part(opt.nclasses, test = True, linear_num=opt.linear_num, size=h)
elif opt.resnet_4head:
    model_structure = ResNet50_4head(opt.nclasses, stride = opt.stride, test = True, linear_num=opt.linear_num, size=h)
elif opt.TwinResNet50:
    model_structure = TwinResNet50(opt.nclasses, stride = opt.stride, test = True, linear_num=opt.linear_num)
else:
    opt.linear_num = 1024
    model_structure = ResNet50(opt.nclasses, test = True, linear_num=opt.linear_num)


#if opt.fp16:
#    model_structure = network_to_half(model_structure)
model = load_network(model_structure)

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

# We can optionally trace the forward method with PyTorch JIT so it runs faster.
# To do so, we can call `.trace` on the reparamtrized module with dummy inputs
# expected by the module.
# Comment out this following line if you do not want to trace.
#dummy_forward_input = torch.rand(opt.batchsize, 3, h, w).cuda()
#model = torch.jit.trace(model, dummy_forward_input)

#print(model)
# Extract feature
since = time.time()
with torch.no_grad():
    gallery_feature = extract_feature(model,dataloaders['gallery'])
    query_feature = extract_feature(model,dataloaders['query'])

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.2f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
# Save to Matlab for check
print(f'Gallery features shape: {gallery_feature.numpy().shape}, label shape: {len(gallery_label)}, cam shape: {len(gallery_cam)} | query features shape: {query_feature.numpy().shape}, label shape: {len(query_label)}, cam shape: {len(query_cam)}')
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
scipy.io.savemat('pytorch_result.mat',result)

print(opt.name)
result = f'./model/{opt.name}/result_{train_type}_{test_type}.txt'
os.system('python3 evaluate_gpu.py | tee -a %s'%result)
os.system('python3 evaluate_rerank.py | tee -a %s'%result)

