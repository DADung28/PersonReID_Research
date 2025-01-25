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
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=128, type=int, help='batchsize')
parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--use_efficient', action='store_true', help='use efficient-b4' )
parser.add_argument('--use_hr', action='store_true', help='use hr18 net' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--use_latrans', action='store_true', help='use LA_Transformer' )
parser.add_argument('--use_vit', action='store_true', help='use Vision Transformer (ViT)' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--ibn', action='store_true', help='use ibn.' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--test',default='market', type=str,help='Choose test dataset type for test')
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
if 'centroid' in config:      
    opt.centroid = config['centroid']        
else:
    opt.centroid = False

opt.fp16 = config['fp16'] 
opt.PCB = config['PCB']
opt.use_dense = config['use_dense']
opt.use_NAS = config['use_NAS']
opt.stride = config['stride']
if 'use_swin' in config:
    opt.use_swin = config['use_swin']
else:
    opt.use_swin = False 
    
if 'use_swinv2' in config:
    opt.use_swinv2 = config['use_swinv2']
else:
    opt.use_swinv2 = False
    
if 'use_convnext' in config:
    opt.use_convnext = config['use_convnext']
else:
    opt.use_convnext = False
    
if 'use_efficient' in config:
    opt.use_efficient = config['use_efficient']
else:
    opt.use_efficient = False
    
if 'use_hr' in config:
    opt.use_hr = config['use_hr']
else:
    opt.use_hr = False
    
if 'use_lenet' in config:
    opt.use_lenet = config['use_lenet']
else:
    opt.use_lenet = False
    
if 'use_vit' in config:
    opt.use_vit = config['use_vit']
else:
    opt.use_vit = False
    
if 'use_vitraw' in config:
    opt.use_vitraw = config['use_vitraw']
else:
    opt.use_vitraw = False
    
if 'use_latrans' in config:
    opt.use_latrans = config['use_latrans']
else:
    opt.use_latrans = False
    
if 'use_latransv2' in config:
    opt.use_latransv2 = config['use_latransv2']
else:
    opt.use_latransv2 = False
    
if 'use_laswin' in config:
    opt.use_laswin = config['use_laswin']
else:
    opt.use_laswin = False
    
if 'use_caswin' in config:
    opt.use_caswin = config['use_caswin']
else:
    opt.use_caswin = False
    
if 'use_laswinv2' in config:
    opt.use_laswinv2 = config['use_laswinv2']
else:
    opt.use_laswinv2 = False
    
if 'ABS' in config:
    opt.ABS = config['ABS']
else:
    opt.ABS = False  

if 'O2LS' in config:
    opt.O2LS = config['O2LS']
else:
    opt.O2LS = False  
    
if 'TripletSwin' in config:
    opt.TripletSwin = config['TripletSwin']
else:
    opt.TripletSwin = False  
    
if 'PCB' in config:
    opt.PCB = config['PCB']
else:
    opt.PCB = False
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

h, w = 224, 224

data_transforms = transforms.Compose([
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])
    h, w = 384, 192


if opt.multi:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query','multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query','multi-query']}
else:
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
    if opt.linear_num <= 0:
        if opt.use_swin or opt.use_swinv2 or opt.use_dense or opt.use_convnext or opt.use_vit or opt.use_vitraw:
            opt.linear_num = 1024
        elif opt.use_efficient:
            opt.linear_num = 1792
        elif opt.use_NAS:
            opt.linear_num = 4032
        
        else:
            opt.linear_num = 2048

    for iter, data in enumerate(dataloaders):
        img, label = data
        n, c, h, w = img.size()
        # count += n
        # print(count)
        pbar.update(n)
        ff = torch.FloatTensor(n,opt.linear_num).zero_().cuda()
        if opt.use_laswinv2 or opt.use_caswin:
            ff = torch.FloatTensor(n,1024*7*7).zero_().cuda() 
        if opt.PCB:
            ff = torch.FloatTensor(n,opt.linear_num*2).zero_().cuda() 
        if opt.use_latrans or opt.use_latransv2:
            ff = torch.FloatTensor(n,768,14).zero_().cuda() # we have 14 parts
            #ff = torch.FloatTensor().cuda() # we have six parts

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
        # norm feature
        #if opt.PCB:
        if False:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
            
        elif opt.use_latrans or opt.use_latransv2:
            #print(f'ff shape: {ff.shape}')
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(14)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
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

if opt.multi:
    mquery_path = image_datasets['multi-query'].imgs
    mquery_cam,mquery_label = get_id(mquery_path)

######################################################################
# Load Collected data Trained model
print('-------test-----------')
return_feature = opt.arcface or opt.cosface or opt.circle or opt.triplet or opt.contrast or opt.instance or opt.lifted or opt.sphere or opt.centroid
if opt.use_dense:
    model_structure = ft_net_dense(opt.nclasses, stride = opt.stride, linear_num=opt.linear_num)
elif opt.use_lenet:
    model_structure = GoogLeNet(opt.nclasses, stride = opt.stride, linear_num=opt.linear_num)
elif opt.use_NAS:
    model_structure = ft_net_NAS(opt.nclasses, linear_num=opt.linear_num)
elif opt.use_swin:
    model_structure = Swin(opt.nclasses, linear_num=opt.linear_num)
elif opt.use_swinv2:
    model_structure = ft_net_swin(opt.nclasses, linear_num=opt.linear_num)
elif opt.use_convnext:
    model_structure = ft_net_convnext(opt.nclasses, linear_num=opt.linear_num)
elif opt.use_efficient:
    model_structure = ft_net_efficient(opt.nclasses, linear_num=opt.linear_num)
elif opt.use_hr:
    model_structure = ft_net_hr(opt.nclasses, linear_num=opt.linear_num)
elif opt.use_latrans or opt.use_latransv2:
    
    model_structure = LATransformerTest(opt.nclasses, circle=False)
elif opt.use_vit:
    model_structure = ViTReID(opt.nclasses, linear_num=opt.linear_num)
elif opt.use_laswin or opt.use_laswinv2:
    model_structure = LASwinV2(opt.nclasses, linear_num=opt.linear_num, circle=return_feature, test=True)
    opt.linear_num = 3584
elif opt.use_caswin:
    model_structure = CASwin(opt.nclasses, linear_num=opt.linear_num, circle=return_feature, test=True)
elif opt.PCB:
    model_structure = PCB_test(opt.nclasses, circle=return_feature, linear_num=opt.linear_num)
elif opt.ABS:
    model_structure = ABSwin(opt.nclasses, linear_num=opt.linear_num, test=True)
elif opt.O2LS:
    #model_structure = O2LSwin_old(opt.nclasses, linear_num=opt.linear_num, test=True, circle=return_feature)
    model_structure = O2LSwin(opt.nclasses, linear_num=opt.linear_num, test=True, circle=True)
    opt.linear_num = 1024
else:
    model_structure = ft_net(opt.nclasses, stride = opt.stride, ibn = opt.ibn, linear_num=opt.linear_num)


#if opt.fp16:
#    model_structure = network_to_half(model_structure)
model = load_network(model_structure)

# Remove the final fc layer and classifier layer
if opt.use_latrans or opt.use_latransv2 or opt.O2LS:
    pass
elif opt.use_laswinv2 or opt.use_caswin or opt.PCB:
    
    if return_feature:
        model.classifier.classifier = nn.Sequential() 
    else: 
        pass
elif opt.ABS:
    model.classifier3.classifier = nn.Sequential()
elif opt.use_laswin:
    pass
else:
    #if opt.fp16:
        #model[1].model.fc = nn.Sequential()
        #model[1].classifier = nn.Sequential()
    #else:
    model.classifier.classifier = nn.Sequential()

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
    if opt.multi:
        mquery_feature = extract_feature(model,dataloaders['multi-query'])
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.2f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
# Save to Matlab for check
print(f'Gallery features shape: {gallery_feature.numpy().shape}, label shape: {len(gallery_label)}, cam shape: {len(gallery_cam)} | query features shape: {query_feature.numpy().shape}, label shape: {len(query_label)}, cam shape: {len(query_cam)} ')
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
scipy.io.savemat('pytorch_result.mat',result)

print(opt.name)
result = f'./model/{opt.name}/result_{train_type}_{test_type}.txt'
os.system('python3 evaluate_gpu.py | tee -a %s'%result)

if opt.multi:
    result = {'mquery_f':mquery_feature.numpy(),'mquery_label':mquery_label,'mquery_cam':mquery_cam}
    scipy.io.savemat('multi_query.mat',result)
