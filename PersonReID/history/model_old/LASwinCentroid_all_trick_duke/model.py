import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import pretrainedmodels
import timm
from utils.utils import load_state_dict_mute
import os

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


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, linear=512, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
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
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x,f]
        else:
            x = self.classifier(x)
            return x

class LASwinV2(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, circle=False, vertical = True, linear_num=512, test=False):
        super().__init__()
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, drop_path_rate = 0.2)
        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.part = 7
        self.patch_embed = model.patch_embed
        self.layers = model.layers
        self.norm = model.norm
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.avgpool1 = nn.MaxPool2d(kernel_size = (2,2), stride = 2)
        self.circle = circle
        self.test = test
        self.relu = nn.ReLU()
        self.vertical = vertical
        #self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.test = test # Test mode
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(1536, class_num, droprate, linear=linear_num, return_f = self.circle))
            
        #if self.circle:
        #    self.avgpool = nn.AdaptiveAvgPool1d(1)
        #    self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f = self.circle) 
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.layers[0](x)
        x = self.layers[1](x)
        x = self.layers[2](x) # torch.Size([32, 14, 14, 512])

        x_2 = x.permute(0,3,1,2) # torch.Size([32, 196, 512])        
        x_2 = self.avgpool1(x_2) # torch.Size([32, 512, 7, 7])
        x_2 = x_2.permute(0,2,3,1) 

        x = self.layers[3](x)
        x = self.norm(x) 
       
        x = torch.cat((x_2, x), dim = 3) # torch.Size([32, 7, 7, 1536]) 
         
        if self.vertical:
            x_part = torch.mean(x, dim=1, keepdim=False) # Slit picture in to 7 path vertically and do average on these parts torch.Size([32, 7, 1024]) 
        else:
            x_part = torch.mean(x, dim=2, keepdim=False) # Slit picture in to 7 path horizontally and do average on these parts torch.Size([32, 7, 1024]) 
        
        x_local = x_part.permute(0,2,1).unsqueeze(dim=3)# torch.Size([32, 1024, 7, 1])
        # print(x_local.shape)
        # Locally aware network
        part = {}
        predict = {}
    
        # get 7 part feature batchsize*1024*7*1
        for i in range(self.part):
            part[i] = x_local[:,:,i].view(x_local.size(0), x_local.size(1))
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])

        y_local = []
        for i in range(self.part):
            y_local.append(predict[i])
        
        if self.test:
            features = y_local[0][1]
            for output in y_local[1:]:
                features = torch.cat((features,output[1]), dim=1) # L3
            #features = torch.cat((x_3[1]), dim=1) # L2+L3
            return features            
        else:
            return y_local
        
class CASwin(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, circle=False, linear_num=512, test=False):
        super().__init__()
        model_ft = timm.create_model('swin_base_patch4_window7_224', pretrained=True, drop_path_rate = 0.2)
        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.head = nn.Sequential() # save memory
        self.model = model_ft
        self.circle = circle
        self.part = 4
        #self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.test = test # Test mode
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(1024, class_num, droprate, linear=linear_num, return_f = False))
                
        if self.circle:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f = self.circle) 
    def forward(self, x):
        x = self.model.forward_features(x)

        x_global = x.view(x.shape[0],-1,x.shape[-1]) # Change shape from [batchsize, 7, 7, 1024] -> [batchsize, 49, 1024]
        
        x_part = torch.mean(x, dim=1, keepdim=False) # Slit picture in to 7 path vertically and do average on these parts torch.Size([32, 7, 1024]) 
        
        if self.circle: # Get the global feature of all
            x_global = self.avgpool(x_global.permute((0,2,1))) # Change shape from [batchsize, 49, 1024] -> [batchsize, 1024, 49] and do average pooling on 2 dimension 
            x_global = x_global.view(x_global.size(0), x_global.size(1))
            y_global = self.classifier(x_global)
        
        
        part_0 = x_part[:,3,:]
        part_1 = x_part[:,2,:] + x_part[:,4,:] / 2
        part_2 = x_part[:,1,:] + x_part[:,5,:] / 2
        part_3 = x_part[:,0,:] + x_part[:,6,:] / 4
        
        x_local = [part_0,part_1,part_2,part_3]
        
        if self.test:
            return x.view(x.shape[0],-1) # Return merged local feature for features extracting torch.Size([32, 50176])
        #print(x_local.shape)
        # Locally aware network
        part = {}
        predict = {}
    
        # get 7 part feature batchsize*1024*7*1
        for i in range(self.part):
            part[i] = x_local[i]
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])

        y_local = []
        for i in range(self.part):
            y_local.append(predict[i])
        if self.circle:
            return [y_global, y_local]
        else:
            return y_local    

class O2LSwin_old(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, circle=False, linear_num=1024, test=False):
        super().__init__()
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, drop_path_rate = 0.2)
        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.patch_embed = model.patch_embed
        self.layers = model.layers
        self.norm = model.norm
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.avgpool1 = nn.AvgPool2d(kernel_size = (2,2), stride = 2)
        self.circle = circle
        self.test = test
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(512, 1024)
        #self.classifier.apply(weights_init_classifier)
        
        self.classifier0 = ClassBlock(128, class_num, droprate, linear=linear_num, return_f = circle)
        self.classifier1 = ClassBlock(256, class_num, droprate, linear=linear_num, return_f = circle)
        self.classifier2 = ClassBlock(512, class_num, droprate, linear=linear_num, return_f = circle)
        self.classifier3 = ClassBlock(1536, class_num, droprate, linear=linear_num, return_f = circle) 
        #print('Make sure timm > 0.6.0 and you can install latest timm version by pip install git+https://github.com/rwightman/pytorch-image-models.git')
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.layers[0](x)
        
        x_0 = x.view(x.shape[0],-1,x.shape[-1]) # torch.Size([32, 196, 512])
        x_0 = self.avgpool(x_0.permute((0,2,1))).squeeze()
        x_0 = self.classifier0(x_0)
        
        x = self.layers[1](x)
        
        x_1 = x.view(x.shape[0],-1,x.shape[-1]) # torch.Size([32, 196, 512])
        x_1 = self.avgpool(x_1.permute((0,2,1))).squeeze()
        x_1 = self.classifier1(x_1)
        
        x = self.layers[2](x) # torch.Size([32, 14, 14, 512])
    
        x_2 = x.permute(0,3,1,2) # torch.Size([32, 196, 512])
        
        x_2 = self.avgpool1(x_2) # torch.Size([32, 512, 7, 7])
        x_2 = x_2.permute(0,2,3,1)
         
        x = self.layers[3](x)
        x = self.norm(x) 
        
        x_3 = torch.cat((x_2, x), dim = 3) # torch.Size([32, 7, 7, 1536]) 
        
        
        x_3 = x_3.view(x_3.shape[0],-1,x_3.shape[-1]) # torch.Size([32, 49, 1536])
        x_3 = self.avgpool(x_3.permute((0,2,1))).squeeze() # torch.Size([32, 1536])
        #x_3 = self.classifier(x_3) 
        
        #print(x_3.shape)
        x_3 = self.classifier3(x_3)
        #print(x_3[0].shape, x_3[1].shape)
        #print(features.shape)
        if self.test:
            features = x_3[1] # L3
            #features = torch.cat((x_3[1]), dim=1) # L2+L3
            return features
        else:
            return [x_3]
  
class O2LSwin(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, circle=False, linear_num=1024, test=False):
        super().__init__()
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, drop_path_rate = 0.2)
        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.patch_embed = model.patch_embed
        self.layers = model.layers
        self.norm = model.norm
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.avgpool1 = nn.MaxPool2d(kernel_size = (2,2), stride = 2)
        self.avgpool2 = nn.AvgPool2d(kernel_size = (5,5), stride = 5)
        self.circle = circle
        self.test = test
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(512, 1024)
        #self.classifier.apply(weights_init_classifier)
        self.conv1 = nn.Conv2d(512,1024,kernel_size=1, stride=1) 
        self.conv2 = nn.Conv2d(2048,2048,kernel_size=3, stride=1) 
        self.classifier0 = ClassBlock(128, class_num, droprate, linear=linear_num, return_f = circle)
        self.classifier1 = ClassBlock(256, class_num, droprate, linear=linear_num, return_f = circle)
        self.classifier2 = ClassBlock(512, class_num, droprate, linear=linear_num, return_f = circle)
        self.classifier3 = ClassBlock(2048, class_num, droprate, linear=linear_num, return_f = circle) 
        #print('Make sure timm > 0.6.0 and you can install latest timm version by pip install git+https://github.com/rwightman/pytorch-image-models.git')
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.layers[0](x)
        
        x_0 = x.view(x.shape[0],-1,x.shape[-1]) # torch.Size([32, 196, 512])
        x_0 = self.avgpool(x_0.permute((0,2,1))).squeeze()
        x_0 = self.classifier0(x_0)
        
        x = self.layers[1](x)
        
        x_1 = x.view(x.shape[0],-1,x.shape[-1]) # torch.Size([32, 196, 512])
        x_1 = self.avgpool(x_1.permute((0,2,1))).squeeze()
        x_1 = self.classifier1(x_1)
        
        x = self.layers[2](x) # torch.Size([32, 14, 14, 512])
    
        x_2 = x.permute(0,3,1,2) # torch.Size([32, 512, 14, 14])
        x_2 = self.conv1(x_2) # torch.Size([32, 1024, 14, 14])
        x_2 = self.avgpool1(x_2) # torch.Size([32, 512, 7, 7])
        x_3 = x_2.relu()
        x_2 = x_2.permute(0,2,3,1)
         
        x = self.layers[3](x)
        x = self.norm(x) 
        
        x_3 = torch.cat((x_2, x), dim = 3) # torch.Size([32, 7, 7, 2048]) 
        x_3 = x_3.permute(0,3,1,2)
        x_3 = self.conv2(x_3)
        x_3 = self.avgpool2(x_3)
        x_3 = self.relu(x_3).squeeze()
        #print(x_3.shape)
        
        #x_3 = x_3.view(x_3.shape[0],-1,x_3.shape[-1]) # torch.Size([32, 49, 2048])
        #x_3 = self.avgpool(x_3.permute((0,2,1))).squeeze() # torch.Size([32, 2048])
        #x_3 = self.classifier(x_3) 
        
        #print(x_3.shape)
        x_3 = self.classifier3(x_3)
        #print(x_3[0].shape, x_3[1].shape)
        #print(features.shape)
        if self.test:
            features = x_3[1] # L3
            #features = torch.cat((x_3[1]), dim=1) # L2+L3
            return features
        else:
            return [x_3]
                
class ABSwin(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, circle=False, linear_num=512, test=False):
        super().__init__()
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, drop_path_rate = 0.2)
        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.patch_embed = model.patch_embed
        self.layers = model.layers
        self.norm = model.norm
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.circle = circle
        self.test = test
        self.classifier0 = ClassBlock(128, class_num, droprate, linear=linear_num, return_f = circle)
        self.classifier1 = ClassBlock(256, class_num, droprate, linear=linear_num, return_f = circle)
        self.classifier2 = ClassBlock(512, class_num, droprate, linear=linear_num, return_f = circle)
        self.classifier3 = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f = circle)
        #print('Make sure timm > 0.6.0 and you can install latest timm version by pip install git+https://github.com/rwightman/pytorch-image-models.git')
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.layers[0](x)
        x_0 = self.avgpool(x.permute(0,3,1,2)).squeeze() # torch.Size([32, 128])
        x = self.layers[1](x)
        x_1 = self.avgpool(x.permute(0,3,1,2)).squeeze() # torch.Size([32, 256])
        x = self.layers[2](x)
        x_2 = self.avgpool(x.permute(0,3,1,2)).squeeze() # torch.Size([32, 512])
        x = self.layers[3](x)
        x = self.norm(x)
        features = self.avgpool(x.permute(0,3,1,2)).squeeze() # torch.Size([32, 1024]) 
        x_0 = self.classifier0(x_0)
        x_1 = self.classifier1(x_1)
        x_2 = self.classifier2(x_2)
        x_3 = self.classifier3(features)
        if self.test:
            return x_3
        else: 
            return [x_0, x_1, x_2, x_3]
    
class LASwin(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, circle=False, linear_num=512, vertical=True, test=False):
        super().__init__()
        model_ft = timm.create_model('swin_base_patch4_window7_224', pretrained=True, drop_path_rate = 0.2)
        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.part = 7
        model_ft.head = nn.Sequential() # save memory
        self.model = model_ft
        self.circle = circle
        #self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.vertical = vertical
        self.test = test # Test mode
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(1024, class_num, droprate, linear=linear_num, return_f = False))
            
        if self.circle:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f = self.circle) 
    def forward(self, x):
        x = self.model.forward_features(x)

        x_global = x.view(x.shape[0],-1,x.shape[-1]) # Change shape from [batchsize, 7, 7, 1024] -> [batchsize, 49, 1024]
        
        if self.vertical:
            x_part = torch.mean(x, dim=1, keepdim=False) # Slit picture in to 7 path vertically and do average on these parts torch.Size([32, 7, 1024]) 
        else:
            x_part = torch.mean(x, dim=2, keepdim=False) # Slit picture in to 7 path horizontally and do average on these parts torch.Size([32, 7, 1024]) 
        
        if self.circle: # Get the global feature of all
            x_global = self.avgpool(x_global.permute((0,2,1))) # Change shape from [batchsize, 49, 1024] -> [batchsize, 1024, 49] and do average pooling on 2 dimension 
            x_global = x_global.view(x_global.size(0), x_global.size(1))
            y_global = self.classifier(x_global)
        
        
        x_local = x_part.permute(0,2,1).unsqueeze(dim=3)# torch.Size([32, 1024, 7, 1])
        if self.test:
            return x.view(x.shape[0],-1) # Return merged local feature for features extracting torch.Size([32, 50176])
        #print(x_local.shape)
        # Locally aware network
        part = {}
        predict = {}
    
        # get 7 part feature batchsize*1024*7*1
        for i in range(self.part):
            part[i] = x_local[:,:,i].view(x_local.size(0), x_local.size(1))
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])

        y_local = []
        for i in range(self.part):
            y_local.append(predict[i])
        if self.circle:
            return [y_global, y_local]
        else:
            return y_local    

class LATransformer(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, stride=2, circle=False, ibn=False, linear_num=256):
        super().__init__()
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=class_num)
        self.class_num = 751
        self.part = 14 # We cut the pool5 to sqrt(N) parts
        self.num_blocks = 12
        self.model = model
        self.model.head.requires_grad_ = False 
        self.cls_token = self.model.cls_token
        self.pos_embed = self.model.pos_embed
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,768))
        self.dropout = nn.Dropout(p=0.5)
        self.lmbd = 8
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(768, class_num, droprate, linear=linear_num, return_f = circle))

    def forward(self,x):
        
        # Divide input image into patch embeddings and add position embeddings
        x = self.model.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) 
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.pos_drop(x + self.pos_embed)
        
        # Feed forward through transformer blocks
        for i in range(self.num_blocks):
            x = self.model.blocks[i](x)
        x = self.model.norm(x)
        
        # extract the cls token
        cls_token_out = x[:, 0].unsqueeze(1)
        G = cls_token_out.expand(-1,196,-1)
        Q = x[:, 1:]
        
        L = (Q + G * self.lmbd)/(1 + self.lmbd)
        
        # Add global cls token to each local token 
        #for i in range(self.part):
        #    out = torch.mul(x[:, i, :], self.lmbd)
        #    x[:,i,:] = torch.div(torch.add(cls_token_out.squeeze(),out), 1+self.lmbd)
        # Average pool
        x = self.avgpool(L)
        # Locally aware network
        part = {}
        predict = {}
        for i in range(self.part):
            part[i] = x[:,i,:]
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])
        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y
    
class LATransformerTest(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, stride=2, circle=False, ibn=False, linear_num=256):
        super().__init__()
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=class_num)
        self.class_num = 751
        self.part = 14 # We cut the pool5 to sqrt(N) parts
        self.num_blocks = 12
        self.model = model
        self.model.head.requires_grad_ = False 
        self.cls_token = self.model.cls_token
        self.pos_embed = self.model.pos_embed
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,768))
        self.dropout = nn.Dropout(p=0.5)
        self.lmbd = 8
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(768, class_num, droprate, linear=linear_num, return_f = circle))
    
    def forward(self,x):
        
        # Divide input image into patch embeddings and add position embeddings
        x = self.model.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) 
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.pos_drop(x + self.pos_embed)
        
        # Feed forward through transformer blocks
        for i in range(self.num_blocks):
            x = self.model.blocks[i](x)
        x = self.model.norm(x)
        
        # extract the cls token
        cls_token_out = x[:, 0].unsqueeze(1)
        
        # Average pool
        x = self.avgpool(x[:, 1:])
        y = x.view(x.size(0),x.size(1),x.size(2)).permute(0,2,1)
        return y


# Define the ViT Model with cls token
class ViTReID(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, stride=2, circle=False, ibn=False, linear_num=512):
        super().__init__()
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=class_num)
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
        self.classifier = ClassBlock(768, class_num, droprate, linear=linear_num, return_f = circle)
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
        x = self.avgpool(x) # torch.Size([32, 197, 768]) 
        x = self.dropout(x)
        x = x.view(x.shape[0], x.shape[2])
        x = self.classifier(x)
        return x
    
# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num=751, droprate=0.5, stride=2, circle=False, ibn=False, linear_num=512):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        if ibn==True:
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, return_f = circle)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


# Define the swin_base_patch4_window7_224 Model
# pytorch > 1.6
class ft_net_swin(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, circle=False, linear_num=512):
        super(ft_net_swin, self).__init__()
        model_ft = timm.create_model('swin_base_patch4_window7_224', pretrained=True, drop_path_rate = 0.2)
        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.head = nn.Sequential() # save memory
        self.model = model_ft
        self.circle = circle
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f = circle)
        #print('Make sure timm > 0.6.0 and you can install latest timm version by pip install git+https://github.com/rwightman/pytorch-image-models.git')
    def forward(self, x):
        x = self.model.forward_features(x)
        x = x.view(x.shape[0],-1,x.shape[-1]) # Change shape from [batchsize, 7, 7, 1024] -> [batchsize, 49, 1024]
        # swin is update in latest timm>0.6.0, so I add the following two lines.
        x = self.avgpool(x.permute((0,2,1))) # Change shape from [batchsize, 49, 1024] -> [batchsize, 1024, 49] and do average pooling on 2 dimension
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

class ft_net_swinv2(nn.Module):

    def __init__(self, class_num, input_size=(256, 128), droprate=0.5, stride=2, circle=False, linear_num=512):
        super(ft_net_swinv2, self).__init__()
        model_ft = timm.create_model('swinv2_base_window8_256', pretrained=True, img_size = input_size, drop_path_rate = 0.2)
        model_full = timm.create_model('swinv2_base_window8_256', pretrained=True)
        load_state_dict_mute(model_ft, model_full.state_dict(), strict=False)
        #model_ft = timm.create_model('swinv2_cr_small_224', pretrained=True, img_size = input_size, drop_path_rate = 0.2)
        # avg pooling to global pooling
        model_ft.head = nn.Sequential() # save memory
        self.model = model_ft
        self.circle = circle
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f = circle)
        #print('Make sure timm > 0.6.0 and you can install latest timm version by pip install git+https://github.com/rwightman/pytorch-image-models.git')
    def forward(self, x):
        x = self.model.forward_features(x)
        x = x.view(x.shape[0],-1,x.shape[-1])
        x = self.avgpool(x.permute((0,2,1))) # B * 1024 * WinNum
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

class ft_net_convnext(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, circle=False, linear_num=512):
        super(ft_net_convnext, self).__init__()
        model_ft = timm.create_model('convnext_base', pretrained=True, drop_path_rate = 0.2)
        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.head = nn.Sequential() # save memory
        self.model = model_ft
        #self.model.apply(activate_drop)
        self.circle = circle
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f = circle)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


# Define the HRNet18-based Model
class ft_net_hr(nn.Module):
    def __init__(self, class_num, droprate=0.5, circle=False, linear_num=512):
        super().__init__()
        model_ft = timm.create_model('hrnet_w18', pretrained=True)
        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.classifier = nn.Sequential() # save memory
        self.model = model_ft
        self.circle = circle
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, return_f = circle)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride = 2, circle=False, linear_num=512):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        if stride == 1:
            model_ft.features.transition3.pool.stride = 1
        self.model = model_ft
        self.circle = circle
        # For DenseNet, the feature dim is 1024 
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f=circle)

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Define the Efficient-b4-based Model
class ft_net_efficient(nn.Module):

    def __init__(self, class_num, droprate=0.5, circle=False, linear_num=512):
        super().__init__()
        #model_ft = timm.create_model('tf_efficientnet_b4', pretrained=True)
        try:
            from efficientnet_pytorch import EfficientNet
        except ImportError:
            print('Please pip install efficientnet_pytorch')
        model_ft = EfficientNet.from_pretrained('efficientnet-b4')
        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.head = nn.Sequential() # save memory
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.classifier = nn.Sequential()
        self.model = model_ft
        self.circle = circle
        # For EfficientNet, the feature dim is not fixed
        # for efficientnet_b2 1408
        # for efficientnet_b4 1792
        self.classifier = ClassBlock(1792, class_num, droprate, linear=linear_num, return_f=circle)
    def forward(self, x):
        #x = self.model.forward_features(x)
        x = self.model.extract_features(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


# Define the NAS-based Model
class ft_net_NAS(nn.Module):

    def __init__(self, class_num, droprate=0.5, linear_num=512):
        super().__init__()  
        model_name = 'nasnetalarge' 
        # pip install pretrainedmodels
        model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.dropout = nn.Sequential()
        model_ft.last_linear = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 4032
        self.classifier = ClassBlock(4032, class_num, droprate, linear=linear_num)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avg_pool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x
    
# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_net_middle(nn.Module):

    def __init__(self, class_num=751, droprate=0.5):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x = self.classifier(x) #use our classifier.
        return x

# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, global_circle=False, local_circle=False ,linear_num=1024):
        super().__init__()

        self.part = 6 # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.global_circle = global_circle
        self.local_circle = local_circle
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, class_num, droprate=0.5, linear=256, relu=False, bnorm=True, return_f = self.local_circle))
            
        if self.global_circle:
            self.avgpool1 = nn.AdaptiveAvgPool1d(1)
            self.classifier = ClassBlock(2048, class_num, droprate=0.5, linear=linear_num, relu=False, bnorm=True, return_f = self.local_circle) 

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x) # torch.Size([32, 2048, 14, 14])
        
        x_global = x.view(x.shape[0], -1, x.shape[1]) # torch.Size([32, 196, 2048])
        if self.global_circle:
            x_global = self.avgpool1(x_global.permute((0,2,1))) # Change shape from [batchsize, 196, 2048] -> [batchsize, 2048, 196] and do average pooling on 2 dimension 
            x_global = x_global.view(x_global.size(0), x_global.size(1))
            y_global = self.classifier(x_global)
        
        x_local = self.avgpool(x) # torch.Size([32, 2048, 6, 1])
        x_local = self.dropout(x_local)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = x_local[:,:,i].view(x.size(0), x_local.size(1))
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])

        # sum prediction
        #y = predict[0]
        #for i in range(self.part-1):
        #    y += predict[i+1]
        y_local = []
        for i in range(self.part):
            y_local.append(predict[i])
            
        if self.global_circle:
            return [y_global, y_local]
        else:
            return y_local

class PCB_test(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, global_circle=False, local_circle=False ,linear_num=1024):
        super().__init__()

        self.part = 6 # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.global_circle = global_circle
        self.local_circle = local_circle
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, class_num, droprate=0.5, linear=256, relu=False, bnorm=True, return_f = self.local_circle))
            
        if self.global_circle:
            self.avgpool1 = nn.AdaptiveAvgPool1d(1)
            self.classifier = ClassBlock(2048, class_num, droprate=0.5, linear=linear_num, relu=False, bnorm=True, return_f = self.local_circle) 

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x) # torch.Size([32, 2048, 14, 14])
        
        x_global = x.view(x.shape[0], -1, x.shape[1]) # torch.Size([32, 196, 2048])
        if self.global_circle:
            x_global = self.avgpool1(x_global.permute((0,2,1))) # Change shape from [batchsize, 196, 2048] -> [batchsize, 2048, 196] and do average pooling on 2 dimension 
            x_global = x_global.view(x_global.size(0), x_global.size(1))
            y_global = self.classifier(x_global)
        
        x_local = self.avgpool(x) # torch.Size([32, 2048, 6, 1])
        x_local = self.dropout(x_local)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = x_local[:,:,i].view(x.size(0), x_local.size(1))
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])

        # sum prediction
        #y = predict[0]
        #for i in range(self.part-1):
        #    y += predict[i+1]
        y_local = []
        for i in range(self.part):
            y_local.append(predict[i])
        
        features = y_global_feature = y_global[1]
        for logits in y_local:
            features = torch.cat((features, logits[1]), dim = 1)
        if self.global_circle:
            return features
        else:
            return y_local
    

'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    net = ft_net_hr(751)
    #net = ft_net_swin(751, stride=1)
    net.classifier = nn.Sequential()
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 224, 224))
    output = net(input)
    print('net output size:')
    print(output.shape)
