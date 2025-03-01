# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from PIL import Image
import time
import os
import collections
from tqdm import tqdm
from model import *
from utils.random_erasing import RandomErasing
from utils.dgfolder import DGFolder
import yaml
from shutil import copyfile
from solver import make_optimizer, make_optimizer_with_center, WarmupMultiStepLR
from utils.circle_loss import CircleLoss, convert_label_to_similarity
from utils.instance_loss import InstanceLoss
from utils.centroid_triplet_loss import CentroidTripletLoss
from online_triplet_loss.losses import *
proxy = 'http://10.0.0.107:3128'
os.environ['http_proxy'] = proxy 
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy
version =  torch.__version__

from pytorch_metric_learning import losses, miners #pip install pytorch-metric-learning

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=int,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='ResNet50', type=str, help='output model name')
# data
parser.add_argument('--data_dir',default='/home/jun/ReID_Dataset/duke/dataloader',type=str, help='training dir path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--DG', action='store_true', help='use extra DG-Market Dataset for training. Please download it from https://github.com/NVlabs/DG-Net#dg-market.' )
parser.add_argument('--h', default=224, type=int, help='input image hight')
parser.add_argument('--w', default=224, type=int, help='input image weight')
# optimizer
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay. More Regularization Smaller Weight.')
parser.add_argument('--total_epoch', default=60, type=int, help='total training epoch')
parser.add_argument('--cosine', action='store_true', help='use cosine lrRate' )
# backbone
parser.add_argument('--linear_num', default=1024, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
parser.add_argument('--stride', default=1, type=int, help='stride')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--semanticRes', action='store_true', help='Use Semantic Res')
parser.add_argument('--semanticVit', action='store_true', help='Use Semantic ViT')
parser.add_argument('--SPResNet50', action='store_true', help='Use Semantic ViT')
parser.add_argument('--TwinViT', action='store_true', help='Use Twin Semantic ViT')
parser.add_argument('--TwinSwin', action='store_true', help='Use Twin Semantic Swin')
parser.add_argument('--resnet', action='store_true', help='use ResNet50 (ViT) with cls token' )
parser.add_argument('--TwinResNet50', action='store_true', help='Use Twin Semantic Swin')
parser.add_argument('--resnet_4head', action='store_true', help='use ResNet50 (ViT) with cls token' )
parser.add_argument('--vit', action='store_true', help='use Vision Transformer (ViT) with cls token' )
parser.add_argument('--swin', action='store_true', help='use Vision Transformer (ViT) with cls token' )
parser.add_argument('--pivit', action='store_true', help='use Vision Transformer (ViT) with cls token' )
parser.add_argument('--pivit_nohead', action='store_true', help='use Vision Transformer (ViT) with cls token' )
parser.add_argument('--pivit_3part', action='store_true', help='use Vision Transformer (ViT) with cls token' )
parser.add_argument('--piswin', action='store_true', help='use Vision Transformer (ViT) with cls token' )
parser.add_argument('--pivit_4part', action='store_true', help='use Vision Transformer (ViT) with cls token' )
parser.add_argument('--piresnet', action='store_true', help='use Vision Transformer (ViT) with cls token' )

# loss
parser.add_argument('--warm_epoch', default=5, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--arcface', action='store_true', help='use ArcFace loss')
parser.add_argument('--centroid', action='store_true', help='use CentroidTripletLoss loss' )
parser.add_argument('--center', action='store_true', help='use Center Loss' )
parser.add_argument('--circle', action='store_true', help='use Circle loss' )
parser.add_argument('--cosface', action='store_true', help='use CosFace loss' )
parser.add_argument('--contrast', action='store_true', help='use contrast loss' )
parser.add_argument('--instance', action='store_true', help='use instance loss' )
parser.add_argument('--ins_gamma', default=32, type=int, help='gamma for instance loss')
parser.add_argument('--triplet', action='store_true', help='use triplet loss' )
parser.add_argument('--lifted', action='store_true', help='use lifted loss' )
parser.add_argument('--sphere', action='store_true', help='use sphere loss' )
parser.add_argument('--adv', default=0.0, type=float, help='use adv loss as 1.0' )
parser.add_argument('--aiter', default=10, type=float, help='use adv loss with iter' )

opt = parser.parse_args()
data_dir = opt.data_dir
name = opt.name 

torch.cuda.set_device(opt.gpu_ids)
cudnn.enabled = True
cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#
h, w = opt.h, opt.w

transform_train_list = [
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((h, w), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((h, w)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_val_list = [
        transforms.Resize(size=(h, w),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.erasing_p>0:
    transform_train_list = transform_train_list +  [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
}



image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                          data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'train_val/val'),
                                          data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=2, pin_memory=True,
                                             prefetch_factor=2, persistent_workers=True) # 8 workers may work faster
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()
since = time.time()
inputs, classes = next(iter(dataloaders['train']))
print(time.time()-since)

######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long().cuda()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    #best_model_wts = model.state_dict()
    #best_acc = 0.0
    warm_up = 0.1 # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['train']/opt.batchsize)*opt.warm_epoch # first 5 epoch
    if opt.arcface:
        criterion_arcface = losses.ArcFaceLoss(num_classes=opt.nclasses, embedding_size=512)
    if opt.cosface: 
        criterion_cosface = losses.CosFaceLoss(num_classes=opt.nclasses, embedding_size=512)
    if opt.circle:
        criterion_circle = CircleLoss(m=0.25, gamma=32) # gamma = 64 may lead to a better result.
    if opt.triplet:
        miner = miners.MultiSimilarityMiner()
        criterion_triplet = losses.TripletMarginLoss(margin=0.3)
    if opt.lifted:
        criterion_lifted = losses.GeneralizedLiftedStructureLoss(neg_margin=1, pos_margin=0)
    if opt.contrast: 
        criterion_contrast = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    if opt.instance:
        criterion_instance = InstanceLoss(gamma = opt.ins_gamma)
    if opt.sphere:
        criterion_sphere = losses.SphereFaceLoss(num_classes=opt.nclasses, embedding_size=512, margin=4)
    if opt.centroid:
        criterion_centroid = CentroidTripletLoss()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
    
            # Phases 'train' and 'val' are visualized in two separate progress bars
            pbar = tqdm()
            pbar.reset(total=len(dataloaders[phase].dataset))
            ordered_dict = collections.OrderedDict(phase="", Loss="", Acc="")

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for iter, data in enumerate(dataloaders[phase]):
                # get the inputs
                inputs, labels = data
                now_batch_size,c,h,w = inputs.shape
                pbar.update(now_batch_size)  # update the pbar even in the last batch
                if now_batch_size<opt.batchsize: # skip the last batch
                    continue
                #print(inputs.shape)
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                # if we use low precision, input also need to be fp16
                #if fp16:
                #    inputs = inputs.half()

                # zero the parameter gradients
            
                # forward
                if phase == 'val':
                    with torch.inference_mode():
                        outputs = model(inputs)   
                else:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                
                if isinstance(outputs[0], list):
                    score = 0.0
                    sm = nn.Softmax(dim=1)
                    loss = 0.0
                    for output in outputs:
                        logits, ff = output
                        score += sm(logits)
                        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                        ff = ff.div(fnorm.expand_as(ff))
                        loss += criterion(logits, labels)
                        if opt.circle: 
                            loss +=  criterion_circle(*convert_label_to_similarity(ff, labels))/now_batch_size
                        if opt.triplet:
                            loss += batch_hard_triplet_loss(labels, ff, margin = 6) #/now_batch_size
                        if opt.centroid:
                            loss += criterion_centroid(ff, labels)  
                    _, preds = torch.max(score.data, 1)
                    
                else:
                    logits, ff = outputs
                    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                    ff = ff.div(fnorm.expand_as(ff))
                    loss = criterion(logits, labels) 
                    #loss = 0
                    _, preds = torch.max(logits.data, 1)
                    
                    if opt.arcface:
                        loss +=  criterion_arcface(ff, labels)/now_batch_size
                    if opt.cosface:
                        loss +=  criterion_cosface(ff, labels)/now_batch_size
                    if opt.circle:
                        loss +=  criterion_circle(*convert_label_to_similarity( ff, labels))/now_batch_size
                    if opt.triplet:
                        hard_pairs = miner(ff, labels)
                        #loss +=  criterion_triplet(ff, labels, hard_pairs) #/now_batch_size
                        loss += batch_hard_triplet_loss(labels, ff, margin = 6) #/now_batch_size
                    if opt.lifted:
                        loss +=  criterion_lifted(ff, labels) #/now_batch_size
                    if opt.contrast:
                        loss +=  criterion_contrast(ff, labels) #/now_batch_size
                    if opt.instance:
                        loss += criterion_instance(ff) /now_batch_size
                    if opt.sphere:
                        loss +=  criterion_sphere(ff, labels)/now_batch_size
                    if opt.centroid:
                        loss += criterion_centroid(ff, labels)

                del inputs
            
                if epoch<opt.warm_epoch and phase == 'train': 
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss = loss*warm_up

                if phase == 'train':
                    
                    loss.backward()
                    optimizer.step()
                
                
                running_loss += loss.item() * now_batch_size
                ordered_dict["Loss"] = f"{loss.item():.4f}"
                del loss
                running_corrects += float(torch.sum(preds == labels.data))
                # Refresh the progress bar in every batch
                ordered_dict["phase"] = phase
                ordered_dict["Acc"] = f"{(float(torch.sum(preds == labels.data)) / now_batch_size):.4f}"
                #ordered_dict["lr"] = scheduler.get_lr()
                pbar.set_postfix(ordered_dict=ordered_dict)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
    
            ordered_dict["phase"] = phase
            ordered_dict["Loss"] = f"{epoch_loss:.4f}"
            ordered_dict["Acc"] = f"{epoch_acc:.4f}"
            pbar.set_postfix(ordered_dict=ordered_dict)
            pbar.close()
            
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)
                       
            # deep copy the model
            if phase == 'val' and epoch%10 == 9:
                last_model_wts = model.state_dict()
                save_network(model, epoch+1)
            
            if phase == 'val':
                draw_curve(epoch)
            if phase == 'train':
                scheduler.step()
                
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')

    return model


######################################################################
# Draw Curve
#---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join('./model',name,'train.jpg'))

######################################################################
# Save model
#---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(opt.gpu_ids)


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

return_feature = opt.arcface or opt.cosface or opt.circle or opt.triplet or opt.contrast or opt.instance or opt.lifted or opt.sphere or opt.centroid

if opt.semanticRes:
    opt.linear_num = 512
    model = ResParsing(len(class_names), opt.droprate, opt.stride, test=False, linear_num=opt.linear_num)
elif opt.semanticVit:
    opt.linear_num =  512
    model = TwinViT_old(len(class_names), opt.droprate, opt.stride, test=False, linear_num=opt.linear_num)
elif opt.TwinViT:
    opt.linear_num =  512
    model = TwinViT(len(class_names), opt.droprate, test=False, linear_num=opt.linear_num, size=h) 
elif opt.TwinSwin:
    opt.linear_num =  512
    model = TwinSwin(len(class_names), opt.droprate, test=False, linear_num=opt.linear_num) 
elif opt.vit:
    opt.linear_num = 512
    model = ViT(len(class_names), opt.droprate, opt.stride, test=False, linear_num=opt.linear_num, size=h)
elif opt.swin:
    opt.linear_num = 512
    model = Swin(len(class_names), opt.droprate,test=False, linear_num=opt.linear_num, size=h)
elif opt.pivit:
    opt.linear_num = 512
    model = PIViT(len(class_names), opt.droprate, test=False, linear_num=opt.linear_num, size=h)
elif opt.pivit_3part:
    opt.linear_num = 512
    model = PIViT_3part(len(class_names), opt.droprate, test=False, linear_num=opt.linear_num)
elif opt.piswin:
    opt.linear_num = 512
    model = PISwin(len(class_names), opt.droprate, test=False, linear_num=opt.linear_num, size=h)
elif opt.pivit_4part:
    opt.linear_num = 512
    model = PIViT_4part(len(class_names), opt.droprate, test=False, linear_num=opt.linear_num, size=h)
elif opt.pivit_nohead:
    opt.linear_num = 512
    model = PIViT_nohead(len(class_names), opt.droprate, test=False, linear_num=opt.linear_num, size=h)
elif opt.piresnet:
    opt.linear_num = 512
    model = PIResNet50(len(class_names), opt.droprate, opt.stride, test=False, linear_num=opt.linear_num)
elif opt.SPResNet50:
    opt.linear_num = 512
    model = SPResNet50(len(class_names), opt.droprate, test=False, linear_num=opt.linear_num)
elif opt.resnet_4head:
    model = ResNet50_4head(len(class_names), opt.droprate, opt.stride, test=False, linear_num=opt.linear_num)
elif opt.TwinResNet50:
    opt.linear_num =  512
    model = TwinResNet50(len(class_names), opt.droprate, test=False, linear_num=opt.linear_num) 
else:    
    model = ResNet50(len(class_names), opt.droprate, opt.stride, test=False, linear_num=opt.linear_num)

opt.nclasses = len(class_names)
print(model)

# model to gpu
model = model.cuda()


#loss_func, center_criterion = make_loss_with_center(cfg, num_classes)  # modified by gu
#optimizer, optimizer_center = make_optimizer_with_center(cfg, model, center_criterion)
#scheduler = WarmupMultiStepLR(optimuzer=optimizer, 
#                              milestones=[40,70], 
#                              gamma=0.1, 
#                              warmup_factor=0.01, 
#                              warmup_iters=10, 
#                              warmup_method='linear')
optim_name = optim.SGD #apex.optimizers.FusedSGD


if opt.semanticVit or opt.TwinViT or opt.TwinSwin or opt.TwinResNet50 or opt.resnet_4head:
    try:
        ignored_params += list(map(id, model.classifier.parameters() ))
    except:
        ignored_params = [] 
    ignored_params += (list(map(id, model.classifier0.parameters() ))
                    +list(map(id, model.classifier1.parameters() ))   
                    )
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    classifier_params = filter(lambda p: id(p) in ignored_params, model.parameters())
    optimizer_ft = optim_name([
            {'params': base_params, 'lr': 0.1*opt.lr},
            {'params': classifier_params, 'lr': opt.lr}
        ], weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)
elif False:
    ignored_params = [] 
    ignored_params += (list(map(id, model.model0.classifier.parameters()))
                    + list(map(id, model.model1.classifier.parameters()))
                    + list(map(id, model.model2.classifier.parameters()))
                    + list(map(id, model.model3.classifier.parameters()))
                    )
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    classifier_params = filter(lambda p: id(p) in ignored_params, model.parameters())
    optimizer_ft = optim_name([
            {'params': base_params, 'lr': 0.1*opt.lr},
            {'params': classifier_params, 'lr': opt.lr}
        ], weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)
else:
    ignored_params = list(map(id, model.classifier.parameters() ))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    classifier_params = model.classifier.parameters()
    optimizer_ft = optim_name([
            {'params': base_params, 'lr': 0.1*opt.lr},
            {'params': classifier_params, 'lr': opt.lr}
        ], weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
#scheduler = WarmupMultiStepLR(optimizer_ft, [40,70], 0.1, 0.01, opt.warm_epoch, 'linear')
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=opt.total_epoch*2//3, gamma=0.1)
if opt.cosine:
    exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, opt.total_epoch, eta_min=0.01*opt.lr)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU. 
#
dir_name = os.path.join('./model',name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
#record every run
copyfile('./train.py', dir_name+'/train.py')
copyfile('./model.py', dir_name+'/model.py')

# save opts
with open('%s/opts.yaml'%dir_name,'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

criterion = nn.CrossEntropyLoss()
model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=opt.total_epoch)

