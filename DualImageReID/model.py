import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.nn import functional as F
from torch.autograd import Variable
from net.pspnet import PSPNet
from hrnet.seg_hrnet import HighResolutionNet, HRNet_2Head
import pretrainedmodels
import timm
from utils.utils import load_state_dict_mute
import os
from torchinfo import summary
from config.default import _C as config
config.defrost()
config.merge_from_file('/home/jun/HRNetReID/config_yml/seg_hrnet_w48.yaml')
config.freeze()
#from .backbones.resnet import BasicBlock, Bottleneck, ResNet
#from .backbones.resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a

proxy = 'http://10.0.0.107:3128'
os.environ['http_proxy'] = proxy 
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy
######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


def activate_drop(m):
    classname = m.__class__.__name__
    if classname.find('Drop') != -1:
        m.p = 0.1
        m.inplace = True


class SegmentationHRNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model = HighResolutionNet(cfg)
        model.init_weights(cfg.MODEL.PRETRAINED)
        self.model = model
        self.interpolate = nn.functional.interpolate
    def forward(self, x):
        x = self.model(x)
        x = self.interpolate(input=x, size=(224, 224), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
        return x


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, linear=512, test = False):
        super().__init__()
        self.test = test
        add_block = []
        if linear>0:
            add_block += [nn.Linear(input_dim, linear)]
        else:
            linear = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(linear)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(linear, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.test:
            return x
        else:
            f = x
            x = self.classifier(x)
            return [x,f]
            

# Define the ViT Model with cls token
class ViT(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, test=False, linear_num=512, size=384):
        super().__init__()
        model = timm.create_model(f'vit_base_patch16_{size}', pretrained=True, num_classes=class_num)
        self.patch_embed = model.patch_embed
        self.pos_drop = model.pos_drop
        self.cls_token = model.cls_token
        self.pos_embed = model.pos_embed
        self.patch_drop = model.patch_drop
        self.norm_pre = model.norm_pre
        self.blocks = model.blocks
        self.norm = model.norm
        self.fc_norm = model.fc_norm
        self.avgpool = nn.AdaptiveAvgPool2d((1,768))
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = ClassBlock(768, class_num, droprate, linear=linear_num, test= test)
    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) 
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x) # torch.Size([32, 197, 768])
        x = self.norm(x) # torch.Size([32, 197, 768])
        x = self.fc_norm(x) # torch.Size([32, 197, 768]) 
        x = self.dropout(x)
        x = self.avgpool(x).squeeze() # torch.Size([32, 197, 768]) 
        x = self.classifier(x)
        return x

# Define the ViT Model with cls token
class PIViT(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, test=False, linear_num=512,size=384):
        super().__init__()
        self.test = test
        #self.conv1 = nn.Conv2d(in_channels=21, out_channels=3, kernel_size=1, stride=1, padding=0)
        parsing_model = SegmentationHRNet(config) # segmentation model
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/hrnetv2_224_lip_six.pth'))
        self.parsing_model = parsing_model
        self.model = ViT(class_num, droprate, test, linear_num, size)
        self.model.classifier = nn.Sequential()
        self.classifier = ClassBlock(768, class_num, droprate, linear=linear_num, test= test)
    def forward(self, x):
        x = F.interpolate(input=x, size=(x.shape[2], x.shape[3]//2), mode='bilinear', align_corners=False) 
        with torch.inference_mode():
            p = self.parsing_model(x)
            p = F.interpolate(input=x, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False) 
        p = p.clone().requires_grad_(True)
        p = F.softmax(p, dim=1)
        background = p[:,0:1,:,:]
        p = torch.cat((background, 1-background),dim = 1)
        #print(x.shape,p.shape)
        # Pass get 0,1 map
        p = torch.argmax(p, dim=1).unsqueeze(1) # torch.Size([16, 2, 28, 28])
        # Pack horizontally
        x = torch.cat((x,x * p), dim = 3)
        x = self.model(x)
        x = self.classifier(x)
        return x

# Define the ViT Model with cls token
class SingleViT(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, test=False, linear_num=512,size=384):
        super().__init__()
        self.test = test
        #self.conv1 = nn.Conv2d(in_channels=21, out_channels=3, kernel_size=1, stride=1, padding=0)
        parsing_model = SegmentationHRNet(config) # segmentation model
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/hrnetv2_224_lip_six.pth'))
        self.parsing_model = parsing_model
        self.model = ViT(class_num, droprate, test, linear_num, size)
        self.model.classifier = nn.Sequential()
        self.classifier = ClassBlock(768, class_num, droprate, linear=linear_num, test= test)
    def forward(self, x):
        x = F.interpolate(input=x, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False) 
        with torch.inference_mode():
            p = self.parsing_model(x)
            p = F.interpolate(input=x, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False) 
        p = p.clone().requires_grad_(True)
        p = F.softmax(p, dim=1)
        background = p[:,0:1,:,:]
        p = torch.cat((background, 1-background),dim = 1)
        #print(x.shape,p.shape)
        # Pass get 0,1 map
        p = torch.argmax(p, dim=1).unsqueeze(1) # torch.Size([16, 2, 28, 28])
        # Pack horizontally
        x = x * p
        x = self.model(x)
        x = self.classifier(x)
        return x


# Define the ViT Model with cls token
class SPViT(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, test=False, linear_num=512,size=384):
        super().__init__()
        self.test = test
        #self.conv1 = nn.Conv2d(in_channels=21, out_channels=3, kernel_size=1, stride=1, padding=0)
        parsing_model = SegmentationHRNet(config) # segmentation model
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/hrnetv2_224_lip_six.pth'))
        self.parsing_model = parsing_model
        self.model = ViT(class_num, droprate, test, linear_num, size)
        self.model.classifier = nn.Sequential()
        self.model.avgpool = nn.Sequential()
        self.avgpool = nn.AdaptiveAvgPool2d((1,768))
        self.classifier0 = ClassBlock(768, class_num, droprate, linear=linear_num, test= test)
        self.classifier1 = ClassBlock(768, class_num, droprate, linear=linear_num, test= test)
        self.classifier2 = ClassBlock(768, class_num, droprate, linear=linear_num, test= test)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, x):
        x_0 = x
        x = self.model(x)
        cls = x[:,0:1,:]
        x = x[:,1:,:]
        x = x.view(x.shape[0],x.shape[2],24,24)
        with torch.inference_mode():
            p = self.parsing_model(x_0)
            p = F.interpolate(input=p, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False) # (16,6,56,56)
        
        x_parsing = p.clone().requires_grad_(True)
        # Get all the filter of head, body, arm, leg and feat
        x_parsing = self.softmax(x_parsing) # torch.Size([16, 5, 28, 28])
        # Get foreground probabilites by do 1-background probabiliry 
        x_foreground = 1 - x_parsing[:,0:1,:,:]# torch.Size([16, 1, 28, 28]) 
        # Replace background propability with foregroiund probability
        x_parsing = torch.cat((x_foreground, x_parsing[:,1:,:,:]), dim=1)
        # l1 normalize through each channel
        x_parsing = F.normalize(x_parsing, p=1, dim=1)
        
        
        x_global = x.view(x.shape[0],-1,x.shape[1])
        x_global = torch.cat((cls,x_global),dim=1)
        x_global = self.avgpool(x_global).squeeze()
        y_global = self.classifier0(x_global)
        
        
        y_foreground = x*x_parsing[:,0:1,:,:]
        y_foreground = y_foreground.view(y_foreground.shape[0],-1,y_foreground.shape[1])
        y_foreground = self.classifier1(self.avgpool(y_foreground).squeeze())
        
        x_part = x*x_parsing[:,1:2,:,:]
        x_part = x_part.view(x_part.shape[0],-1,x_part.shape[1])
        y_part = self.avgpool(x_part).squeeze()
        for i in range(2,x_parsing.shape[1]):
            x_part = x*x_parsing[:,i:i+1,:,:]
            x_part = x_part.view(x_part.shape[0],-1,x_part.shape[1])
            y_part += self.avgpool(x_part).squeeze()
        y_part = self.classifier1(y_part)
        
        y = [y_global, y_foreground, y_part]
        if self.test:
            return torch.cat(y, dim=1)
        else:
            return y
        
        
        

# Define the ViT Model with cls token
class PIViT_weak(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, test=False, linear_num=512,size=384):
        super().__init__()
        self.test = test
        #self.conv1 = nn.Conv2d(in_channels=21, out_channels=3, kernel_size=1, stride=1, padding=0)
        parsing_model = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 2, deep_features_size=512)
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/binary_parsing.pth'))
        self.parsing_model = parsing_model
        self.model = ViT(class_num, droprate, test, linear_num, size)
        self.model.classifier = nn.Sequential()
        self.classifier = ClassBlock(768, class_num, droprate, linear=linear_num, test= test)
    def forward(self, x):
        x = F.interpolate(input=x, size=(x.shape[2], x.shape[3]//2), mode='bilinear', align_corners=False) 
        with torch.inference_mode():
            p , class_score= self.parsing_model(x)
        p = p.clone().requires_grad_(True)
        # Pass get 0,1 map
        p = torch.argmax(p, dim=1).unsqueeze(1) # torch.Size([16, 2, 28, 28])
        # Pack horizontally
        x = torch.cat((x,x * p), dim = 3)
        x = self.model(x)
        x = self.classifier(x)
        return x
    
# Define the ViT Model with cls token
class PIViT_3loss(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, test=False, linear_num=512,size=384):
        super().__init__()
        self.test = test
        #self.conv1 = nn.Conv2d(in_channels=21, out_channels=3, kernel_size=1, stride=1, padding=0)
        parsing_model = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 2, deep_features_size=512)
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/binary_parsing.pth'))
        self.parsing_model = parsing_model
        self.model = ViT(class_num, droprate, test, linear_num, size)
        self.model.avgpool = nn.Sequential()
        self.model.classifier = nn.Sequential()
        self.avgpool = nn.AdaptiveAvgPool2d((1,768))
        self.classifier0 = ClassBlock(768, class_num, droprate, linear=linear_num, test= test)
        self.classifier1 = ClassBlock(768, class_num, droprate, linear=linear_num, test= test)
        self.classifier2 = ClassBlock(768, class_num, droprate, linear=linear_num, test= test)
    
    def forward(self, x):
        x = F.interpolate(input=x, size=(x.shape[2], x.shape[3]//2), mode='bilinear', align_corners=False) 
        with torch.inference_mode():
            p , class_score= self.parsing_model(x)
        p = p.clone().requires_grad_(True)
        # Pass get 0,1 map
        p = torch.argmax(p, dim=1).unsqueeze(1) # torch.Size([16, 2, 28, 28])
        # Pack horizontally
        x = torch.cat((x,x * p), dim = 3)
        x = self.model(x)
        y_global = self.avgpool(x).squeeze()
        y_global = self.classifier0(y_global)
       
        cls = x[:,0:1,:]
        x_global = x[:,1:,:].view(x.shape[0],14,14,x.shape[2])
        
        
        x_left = x_global[:,:,:7,:]
        x_left = x_left.reshape(x_left.shape[0], -1, x_left.shape[3])
        x_left = torch.cat((cls,x_left), dim=1)
        y_left = self.avgpool(x_left).squeeze()
        y_left = self.classifier1(y_left)
       
        x_right = x_global[:,:,7:,:]
        x_right = x_right.reshape(x_right.shape[0], -1, x_right.shape[3])
        x_right = torch.cat((cls,x_right), dim=1)
        y_right = self.avgpool(x_right).squeeze()
        y_right = self.classifier2(y_right)
        
        
        y = [y_global, y_left, y_right]
        
        if self.test:
            return torch.cat(y, dim=1)
        else:
            return y

    
class Swin(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, test=False, linear_num=512, size=384):
        super().__init__()
        if size == 384:
            model_ft = timm.create_model('swin_base_patch4_window12_384', pretrained=True, drop_path_rate = 0.2)
        else:
            model_ft = timm.create_model('swin_base_patch4_window7_224', pretrained=True, drop_path_rate = 0.2)
        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.head = nn.Sequential() # save memory
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, test= test)
    def forward(self, x):
        x = self.model.forward_features(x)
        x = x.view(x.shape[0],-1,x.shape[-1]) # Change shape from [batchsize, 7, 7, 1024] -> [batchsize, 49, 1024]
        # swin is update in latest timm>0.6.0, so I add the following two lines.
        x = self.avgpool(x.permute((0,2,1))).squeeze() # Change shape from [batchsize, 49, 1024] -> [batchsize, 1024, 49] and do average pooling on 2 dimension
        x = self.classifier(x)
        return x 

# Define the ViT Model with cls token
class PISwin_old(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, test=False, linear_num=512, size=384):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.test = test
        #self.conv1 = nn.Conv2d(in_channels=21, out_channels=3, kernel_size=1, stride=1, padding=0)
        parsing_model = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 2, deep_features_size=512)
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/binary_parsing.pth'))
        self.parsing_model = parsing_model
        self.model = Swin(class_num, droprate, test, linear_num, size)
        self.model.classifier = nn.Sequential()
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, test= test)
    def forward(self, x):
        x = F.interpolate(input=x, size=(x.shape[2], x.shape[3]//2), mode='bilinear', align_corners=False) 
        with torch.inference_mode():
            p , class_score= self.parsing_model(x)
        p = p.clone().requires_grad_(True)
        # Pass get 0,1 map
        p = torch.argmax(p, dim=1).unsqueeze(1) # torch.Size([16, 2, 28, 28])
        # Pack horizontally
        x = torch.cat((x,x * p), dim = 3)
        x = self.model(x)
        x = self.classifier(x)
        return x

# Define the ViT Model with cls token
class PISwin(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, test=False, linear_num=512, size=384):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.test = test
        #self.conv1 = nn.Conv2d(in_channels=21, out_channels=3, kernel_size=1, stride=1, padding=0)
        parsing_model = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 2, deep_features_size=512)
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/binary_parsing.pth'))
        self.parsing_model = parsing_model
        self.model = Swin(class_num, droprate, test, linear_num, size)
        self.model.classifier = nn.Sequential()
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, test= test)
    def forward(self, x):
        x = F.interpolate(input=x, size=(x.shape[2], x.shape[3]//2), mode='bilinear', align_corners=False) 
        with torch.inference_mode():
            p = self.parsing_model(x)
            p = F.interpolate(input=x, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False) 
        p = p.clone().requires_grad_(True)
        p = F.softmax(p, dim=1)
        background = p[:,0:1,:,:]
        p = torch.cat((background, 1-background),dim = 1)
        #print(x.shape,p.shape)
        # Pass get 0,1 map
        p = torch.argmax(p, dim=1).unsqueeze(1) # torch.Size([16, 2, 28, 28])
        # Pack horizontally
        x = torch.cat((x,x * p), dim = 3)
        x = self.model(x)
        x = self.classifier(x)
        return x

        
# Define the ViT Model with cls token
class SingleSwin(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, test=False, linear_num=512, size=384):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.test = test
        #self.conv1 = nn.Conv2d(in_channels=21, out_channels=3, kernel_size=1, stride=1, padding=0)
        parsing_model = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 2, deep_features_size=512)
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/binary_parsing.pth'))
        self.parsing_model = parsing_model
        self.model = Swin(class_num, droprate, test, linear_num, size)
        self.model.classifier = nn.Sequential()
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, test= test)
    def forward(self, x):
        x = F.interpolate(input=x, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False) 
        with torch.inference_mode():
            p = self.parsing_model(x)
            p = F.interpolate(input=x, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False) 
        p = p.clone().requires_grad_(True)
        p = F.softmax(p, dim=1)
        background = p[:,0:1,:,:]
        p = torch.cat((background, 1-background),dim = 1)
        #print(x.shape,p.shape)
        # Pass get 0,1 map
        p = torch.argmax(p, dim=1).unsqueeze(1) # torch.Size([16, 2, 28, 28])
        # Pack horizontally
        x = x * p
        x = self.model(x)
        x = self.classifier(x)
        return x
    
# Define the ViT Model with cls token
class SPSwin(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, test=False, linear_num=512, size=384):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.test = test
        #self.conv1 = nn.Conv2d(in_channels=21, out_channels=3, kernel_size=1, stride=1, padding=0)
        parsing_model = SegmentationHRNet(config) # segmentation model
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/hrnetv2_224_lip_six.pth'))
        self.parsing_model = parsing_model
        self.model = Swin(class_num, droprate, test, linear_num, size)
        self.model.classifier = nn.Sequential()
        self.model.avgpool = nn.Sequential()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier0 = ClassBlock(1024, class_num, droprate, linear=linear_num, test= test)
        self.classifier1 = ClassBlock(1024, class_num, droprate, linear=linear_num, test= test)
        self.classifier2 = ClassBlock(1024, class_num, droprate, linear=linear_num, test= test)
    def forward(self, x):
        x_0 = x
        x = self.model(x)
        x = x.view(x.shape[0],x.shape[1],12,12)
        with torch.inference_mode():
            p = self.parsing_model(x_0)
            p = F.interpolate(input=p, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False) # (16,6,56,56)
        
        x_parsing = p.clone().requires_grad_(True)
        # Get all the filter of head, body, arm, leg and feat
        x_parsing = self.softmax(x_parsing) # torch.Size([16, 5, 28, 28])
        # Get foreground probabilites by do 1-background probabiliry 
        x_foreground = 1 - x_parsing[:,0:1,:,:]# torch.Size([16, 1, 28, 28]) 
        # Replace background propability with foregroiund probability
        x_parsing = torch.cat((x_foreground, x_parsing[:,1:,:,:]), dim=1)
        # l1 normalize through each channel
        x_parsing = F.normalize(x_parsing, p=1, dim=1)
        
        x_global = self.avgpool(x).squeeze()
        y_global = self.classifier0(x_global)
        
        
        y_foreground = x*x_parsing[:,0:1,:,:]
        y_foreground = self.classifier1(self.avgpool(y_foreground).squeeze())
        y_part = self.avgpool(x*x_parsing[:,1:2,:,:]).squeeze()
        for i in range(2,x_parsing.shape[1]):
            y_part += self.avgpool(x*x_parsing[:,i:i+1,:,:]).squeeze()
        y_part = self.classifier1(y_part)
        
        y = [y_global, y_foreground, y_part]
        if self.test:
            return torch.cat(y, dim=1)
        else:
            return y

# Define the ViT Model with cls token
class PISwin_3loss(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, test=False, linear_num=512, size=384):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.test = test
        #self.conv1 = nn.Conv2d(in_channels=21, out_channels=3, kernel_size=1, stride=1, padding=0)
        parsing_model = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 2, deep_features_size=512)
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/binary_parsing.pth'))
        self.parsing_model = parsing_model
        self.model = Swin(class_num, droprate, test, linear_num, size)
        self.model.avgpool = nn.Sequential()
        self.model.classifier = nn.Sequential()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier0 = ClassBlock(1024, class_num, droprate, linear=linear_num, test= test)
        self.classifier1 = ClassBlock(1024, class_num, droprate, linear=linear_num, test= test)
        self.classifier2 = ClassBlock(1024, class_num, droprate, linear=linear_num, test= test)
    def forward(self, x):
        x = F.interpolate(input=x, size=(x.shape[2], x.shape[3]//2), mode='bilinear', align_corners=False) 
        with torch.inference_mode():
            p , class_score= self.parsing_model(x)
        p = p.clone().requires_grad_(True)
        # Pass get 0,1 map
        p = torch.argmax(p, dim=1).unsqueeze(1) # torch.Size([16, 2, 28, 28])
        # Get all the filter of head, body, arm, leg and feat
       # Pack horizontally
        x = torch.cat((x,x * p), dim = 3)
        x = self.model(x)
        x = x.view(x.shape[0],x.shape[1],12,12)
        x_right = x[:,:,:,:6]
        x_left = x[:,:,:,6:]
        x_global = self.avgpool(x).squeeze()
        x_left = self.avgpool(x_left).squeeze()
        x_right = self.avgpool(x_right).squeeze()
        y_global = self.classifier0(x_global)
        y_left = self.classifier1(x_left)
        y_right = self.classifier2(x_right)
        y = [y_global, y_left, y_right]
        if self.test:
            return torch.cat(y, dim=1)
        else:
            return y
 
# Define the ResNet50-based Model
class ResNet50(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, stride=1, test=False, linear_num=512):
        super().__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, test= test)
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x).squeeze()
        x = self.classifier(x)
        return x 
    
# Define the ViT Model with cls token
class PIResNet50(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, stride=2, test=False, linear_num=512):
        super().__init__()
        self.test = test
        #self.conv1 = nn.Conv2d(in_channels=21, out_channels=3, kernel_size=1, stride=1, padding=0)
        parsing_model = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 2, deep_features_size=512)
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/binary_parsing.pth'))
        self.parsing_model = parsing_model
        self.model = ResNet50(class_num, droprate, stride, test, linear_num)
        self.model.classifier = nn.Sequential()
        self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, test= test)
    def forward(self, x):
        x = F.interpolate(input=x, size=(x.shape[2], x.shape[3]//2), mode='bilinear', align_corners=False) 
        with torch.inference_mode():
            p , class_score= self.parsing_model(x)
        p = p.clone().requires_grad_(True)
        # Pass get 0,1 map
        p = torch.argmax(p, dim=1).unsqueeze(1) # torch.Size([16, 2, 28, 28])
        # Get all the filter of head, body, arm, leg and feat
       # Pack horizontally
        x = torch.cat((x,x * p), dim = 3)
        x = self.model(x)
        x = self.classifier(x)
        return x

# Define the ViT Model with cls token
class SingleResNet50(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, stride=2, test=False, linear_num=512):
        super().__init__()
        self.test = test
        #self.conv1 = nn.Conv2d(in_channels=21, out_channels=3, kernel_size=1, stride=1, padding=0)
        parsing_model = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 2, deep_features_size=512)
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/binary_parsing.pth'))
        self.parsing_model = parsing_model
        self.model = ResNet50(class_num, droprate, stride, test, linear_num)
        self.model.classifier = nn.Sequential()
        self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, test= test)
    def forward(self, x):
        x = F.interpolate(input=x, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False) 
        with torch.inference_mode():
            p , class_score= self.parsing_model(x)
        p = p.clone().requires_grad_(True)
        # Pass get 0,1 map
        p = torch.argmax(p, dim=1).unsqueeze(1) # torch.Size([16, 2, 28, 28])
        # Get all the filter of head, body, arm, leg and feat
       # Pack horizontally
        x = x * p
        x = self.model(x)
        x = self.classifier(x)
        return x
    
# Define the ViT Model with cls token
class SPResNet50(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, stride=1, test=False, linear_num=512):
        super().__init__()
        self.test = test
        #self.conv1 = nn.Conv2d(in_channels=21, out_channels=3, kernel_size=1, stride=1, padding=0)
        parsing_model = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 6, deep_features_size=512)
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/six_parsing.pth'))
        self.parsing_model = parsing_model
        self.softmax = torch.nn.Softmax(dim=1)
        self.model = ResNet50(class_num, droprate, stride, test, linear_num)
        self.model.model.avgpool = nn.Sequential()
        self.model.classifier = nn.Sequential()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier0 = ClassBlock(2048, class_num, droprate, linear=linear_num, test= test)
        self.classifier1 = ClassBlock(2048, class_num, droprate, linear=linear_num, test= test)
        self.classifier2 = ClassBlock(2048, class_num, droprate, linear=linear_num, test= test)
    def forward(self, x):
        x_0 = x
        x = self.model(x)
        
        with torch.inference_mode():
            p , class_score= self.parsing_model(x_0)
            p = F.interpolate(input=p, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False) # (16,6,56,56)
        x_parsing = p.clone().requires_grad_(True)
        # Get all the filter of head, body, arm, leg and feat
        x_parsing = self.softmax(x_parsing) # torch.Size([16, 5, 28, 28])
        # Get foreground probabilites by do 1-background probabiliry 
        x_foreground = 1 - x_parsing[:,0:1,:,:]# torch.Size([16, 1, 28, 28]) 
        # Replace background propability with foregroiund probability
        x_parsing = torch.cat((x_foreground, x_parsing[:,1:,:,:]), dim=1)
        # l1 normalize through each channel
        x_parsing = F.normalize(x_parsing, p=1, dim=1)
        
        x_global = self.avgpool(x).squeeze()
        y_global = self.classifier0(x_global)
        
        x = x.view(x.shape[0],x.shape[1],-1)
        x_parsing = x_parsing.view(x_parsing.shape[0],x_parsing.shape[1],-1) 
        
        x_parsing = x_parsing.permute(0,2,1)
        
        x_filted = torch.matmul(x,x_parsing)
        x_foreground = x_filted[:,:,0:1].squeeze()
        y_foreground = self.classifier1(x_foreground)
        x_part = torch.sum(x_filted[:,:,1:],dim=2).squeeze()
        y_part = self.classifier2(x_part)
        y = [y_global, y_foreground, y_part]
        if self.test:
            return torch.cat(y, dim=1)
        else:
            return y
    
class SPResNet50_v2(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, stride=1, test=False, linear_num=512):
        super().__init__()
        self.test = test
        #self.conv1 = nn.Conv2d(in_channels=21, out_channels=3, kernel_size=1, stride=1, padding=0)
        parsing_model = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 6, deep_features_size=512)
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/six_parsing.pth'))
        self.parsing_model = parsing_model
        self.softmax = torch.nn.Softmax(dim=1)
        self.model = ResNet50(class_num, droprate, stride, test, linear_num)
        self.model.model.avgpool = nn.Sequential()
        self.model.classifier = nn.Sequential()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier0 = ClassBlock(2048, class_num, droprate, linear=linear_num, test= test)
        self.classifier1 = ClassBlock(2048, class_num, droprate, linear=linear_num, test= test)
        self.classifier2 = ClassBlock(2048, class_num, droprate, linear=linear_num, test= test)
    def forward(self, x):
        x_0 = x
        x = self.model(x)
        
        with torch.inference_mode():
            p , class_score= self.parsing_model(x_0)
            p = F.interpolate(input=p, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False) # (16,6,56,56)
        x_parsing = p.clone().requires_grad_(True)
        # Get all the filter of head, body, arm, leg and feat
        x_parsing = self.softmax(x_parsing) # torch.Size([16, 5, 28, 28])
        # Get foreground probabilites by do 1-background probabiliry 
        x_foreground = 1 - x_parsing[:,0:1,:,:]# torch.Size([16, 1, 28, 28]) 
        # Replace background propability with foregroiund probability
        x_parsing = torch.cat((x_foreground, x_parsing[:,1:,:,:]), dim=1)
        # l1 normalize through each channel
        x_parsing = F.normalize(x_parsing, p=1, dim=1)
        
        x_global = self.avgpool(x).squeeze()
        y_global = self.classifier0(x_global)
        
        
        y_foreground = x*x_parsing[:,0:1,:,:]
        y_foreground = self.classifier1(self.avgpool(y_foreground).squeeze())
        y_part = self.avgpool(x*x_parsing[:,1:2,:,:]).squeeze()
        for i in range(2,x_parsing.shape[1]):
            y_part += self.avgpool(x*x_parsing[:,i:i+1,:,:]).squeeze()
        y_part = self.classifier1(self.avgpool(y_part).squeeze())
            
        y = [y_global, y_foreground, y_part]
        if self.test:
            return torch.cat(y, dim=1)
        else:
            return y
    
# Define the ViT Model with cls token
class PIResNet50_3loss(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, stride=2, test=False, linear_num=512):
        super().__init__()
        self.test = test
        #self.conv1 = nn.Conv2d(in_channels=21, out_channels=3, kernel_size=1, stride=1, padding=0)
        parsing_model = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 2, deep_features_size=512)
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/binary_parsing.pth'))
        self.parsing_model = parsing_model
        self.model = ResNet50(class_num, droprate, stride, test, linear_num)
        self.model.classifier = nn.Sequential()
        self.model.model.avgpool = nn.Sequential()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier0 = ClassBlock(2048, class_num, droprate, linear=linear_num, test= test)
        self.classifier1 = ClassBlock(2048, class_num, droprate, linear=linear_num, test= test)
        self.classifier2 = ClassBlock(2048, class_num, droprate, linear=linear_num, test= test)
    def forward(self, x):
        x = F.interpolate(input=x, size=(x.shape[2], x.shape[3]//2), mode='bilinear', align_corners=False) 
        with torch.inference_mode():
            p , class_score= self.parsing_model(x)
        p = p.clone().requires_grad_(True)
        # Pass get 0,1 map
        p = torch.argmax(p, dim=1).unsqueeze(1) # torch.Size([16, 2, 28, 28])
        # Get all the filter of head, body, arm, leg and feat
       # Pack horizontally
        x = torch.cat((x,x * p), dim = 3)
        x = self.model(x)
        x_right = x[:,:,:,:4]
        x_left = x[:,:,:,4:]
        x_global = self.avgpool(x).squeeze()
        x_left = self.avgpool(x_left).squeeze()
        x_right = self.avgpool(x_right).squeeze()
        #print(x_global.shape, x_left.shape, x_right.shape)
        y_global = self.classifier0(x_global)
        y_left = self.classifier1(x_left)
        y_right = self.classifier2(x_right)
        y = [y_global, y_left, y_right]
        if self.test:
            return torch.cat(y, dim=1)
        else:
            return y

# Define the HRNet18-based Model
class HRNet(nn.Module):
    def __init__(self, class_num, droprate=0.5, test=False, linear_num=512):
        super().__init__()
        model_ft = timm.create_model('hrnet_w32', pretrained=True)
        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.classifier = nn.Sequential() # save memory
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, test=test)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.avgpool(x).squeeze()
        x = self.classifier(x)
        return x

# Define the ViT Model with cls token
class PIHRNet(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, test=False, linear_num=512):
        super().__init__()
        self.test = test
        #self.conv1 = nn.Conv2d(in_channels=21, out_channels=3, kernel_size=1, stride=1, padding=0)
        parsing_model = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 2, deep_features_size=512)
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/binary_parsing.pth'))
        self.parsing_model = parsing_model
        self.model = HRNet(class_num, droprate, test, linear_num)
        self.model.classifier = nn.Sequential()
        self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, test= test)
    def forward(self, x):
        
        x = F.interpolate(input=x, size=(x.shape[2], x.shape[3]//2), mode='bilinear', align_corners=False) 
        with torch.inference_mode():
            p , class_score= self.parsing_model(x)
        p = p.clone().requires_grad_(True)
        # Pass get 0,1 map
        p = torch.argmax(p, dim=1).unsqueeze(1) # torch.Size([16, 2, 28, 28])
        # Get all the filter of head, body, arm, leg and feat
       # Pack horizontally
        x = torch.cat((x,x * p), dim = 3)
        x = self.model(x)
        
        x = self.classifier(x)
        return x

# Define the ViT Model with cls token
class SingleHRNet(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, test=False, linear_num=512):
        super().__init__()
        self.test = test
        #self.conv1 = nn.Conv2d(in_channels=21, out_channels=3, kernel_size=1, stride=1, padding=0)
        parsing_model = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 2, deep_features_size=512)
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/binary_parsing.pth'))
        self.parsing_model = parsing_model
        self.model = HRNet(class_num, droprate, test, linear_num)
        self.model.classifier = nn.Sequential()
        self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, test= test)
    def forward(self, x):
        
        x = F.interpolate(input=x, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False) 
        with torch.inference_mode():
            p , class_score= self.parsing_model(x)
        p = p.clone().requires_grad_(True)
        # Pass get 0,1 map
        p = torch.argmax(p, dim=1).unsqueeze(1) # torch.Size([16, 2, 28, 28])
        # Get all the filter of head, body, arm, leg and feat
       # Pack horizontally
        x = x * p
        x = self.model(x)
        
        x = self.classifier(x)
        return x
    
# Define the ViT Model with cls token
class SPHRNet(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, test=False, linear_num=512):
        super().__init__()
        self.test = test
        #self.conv1 = nn.Conv2d(in_channels=21, out_channels=3, kernel_size=1, stride=1, padding=0)
        parsing_model = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 6, deep_features_size=512)
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/six_parsing.pth'))
        self.parsing_model = parsing_model
        self.softmax = torch.nn.Softmax(dim=1)
        self.model = HRNet(class_num, droprate, test, linear_num)
        self.model.avgpool = nn.Sequential()
        self.model.classifier = nn.Sequential()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier0 = ClassBlock(2048, class_num, droprate, linear=linear_num, test= test)
        self.classifier1 = ClassBlock(2048, class_num, droprate, linear=linear_num, test= test)
        self.classifier2 = ClassBlock(2048, class_num, droprate, linear=linear_num, test= test)
    def forward(self, x):
        x_0 = x
        x = self.model(x)
        
        with torch.inference_mode():
            p , class_score= self.parsing_model(x_0)
            p = F.interpolate(input=p, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False) # (16,6,56,56)
        x_parsing = p.clone().requires_grad_(True)
        # Get all the filter of head, body, arm, leg and feat
        x_parsing = self.softmax(x_parsing) # torch.Size([16, 5, 28, 28])
        # Get foreground probabilites by do 1-background probabiliry 
        x_foreground = 1 - x_parsing[:,0:1,:,:]# torch.Size([16, 1, 28, 28]) 
        # Replace background propability with foregroiund probability
        x_parsing = torch.cat((x_foreground, x_parsing[:,1:,:,:]), dim=1)
        # l1 normalize through each channel
        x_parsing = F.normalize(x_parsing, p=1, dim=1)
        
        x_global = self.avgpool(x).squeeze()
        y_global = self.classifier0(x_global)
        
        x = x.view(x.shape[0],x.shape[1],-1)
        x_parsing = x_parsing.view(x_parsing.shape[0],x_parsing.shape[1],-1) 
        
        x_parsing = x_parsing.permute(0,2,1)
        
        x_filted = torch.matmul(x,x_parsing)
        x_foreground = x_filted[:,:,0:1].squeeze()
        y_foreground = self.classifier1(x_foreground)
        x_part = torch.sum(x_filted[:,:,1:],dim=2).squeeze()
        y_part = self.classifier2(x_part)
        y = [y_global, y_foreground, y_part]
        if self.test:
            return torch.cat(y, dim=1)
        else:
            return y


'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    #net = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=1)
    net = SingleResNet50()
    #net = ft_net_swin(751, stride=1)
    #summary(model=net, 
    #    input_size=(32, 3, 384, 384), # make sure this is "input_size", not "input_shape"
    #    # col_names=["input_size"], # uncomment for smaller output
    #    col_names=["input_size", "output_size", "num_params", "trainable"],
    #    col_width=20,
    #   row_settings=["var_names"]
    #)
    net = net.to('cpu')
    input = Variable(torch.FloatTensor(8, 3, 384, 384)).to('cpu')
    outputs = net(input)
    print('net output size:')
    if isinstance(outputs, list):
        
        for i, output in enumerate(outputs): # If multiple output
            if isinstance(output, list): # If this model have multiple branch
                ff,logit=output
                print(f'Branch {i}: Shape of id tensor {ff.shape}, shape of score tensor{logit.shape}')
            else:
                ff,logit = outputs[0], outputs[1]
                if i==0:
                    ff = output    
                    print(f'This model have only one brach: Shape of id tensor {ff.shape}, shape of score tensor{logit.shape}') # If this model have one branch
    else:
        print(outputs.shape)