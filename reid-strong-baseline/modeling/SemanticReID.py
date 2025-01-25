# Define the ResNet50-based Model
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torchvision import models
from torch.autograd import Variable
import pretrainedmodels
import timm
import os
from .net.pspnet import PSPNet
from torchinfo import summary
proxy = 'http://10.0.0.107:3128'
os.environ['http_proxy'] = proxy 
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class ClassBlock(nn.Module):
    def __init__(self, num_classes, neck, neck_feat, in_planes):
        super().__init__()
        add_block = []
        self.neck = neck
        self.neck_feat = neck_feat
        if self.neck == 'no':
            self.classifier = nn.Linear(in_planes, num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(in_planes, num_classes, bias=False)
            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)
            
    def forward(self, x):
        if self.neck == 'no':
            feat = x
        elif self.neck == 'bnneck':
            feat = self.bottleneck(x)  # normalize for angular softmax
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, x  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return x


class SPResNet50_IMPROVED(nn.Module):
    def __init__(self, num_classes, last_stride, neck, neck_feat):
        super().__init__()
        parsing_model = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 6, deep_features_size=512)
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/six_parsing.pth'))
        self.parsing_model = parsing_model
        self.softmax = torch.nn.Softmax(dim=1)
        model_ft = models.resnet50(pretrained=True)
        if last_stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
        self.model = model_ft 
        in_planes = 2048
        self.classifier = ClassBlock(num_classes, neck, neck_feat, in_planes)
    
    def forward(self, x):
        x_0 = x
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
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
         
        #x = torch.matmul(x, f_copy) # (16, 2048, 20)
        y = []
        x_global = self.model.avgpool(x).squeeze()
        y_global = self.classifier(x_global)
        y.append(y_global)
        
        for i in range(x_parsing.shape[1]):
            mask = x_parsing[:,i:i+1,:]
            x_part = self.model.avgpool(torch.mul(x, mask)).squeeze() # (16, 2048)
            y_part = self.classifier(x_part)
            y.append(y_part) 
        
        if self.training:
            return tuple(y)
        else:
            return torch.cat(y, dim=1)

        
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    net = SPResNet50_21_loss(702, 1, 'bnneck', 'after').to(0)
    #net = ft_net_swin(751, stride=1)
    print(net)
    input = torch.FloatTensor(8, 3, 224, 224).to(0)
    net.train()
    #summary(model=net, 
    #    input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
    #    # col_names=["input_size"], # uncomment for smaller output
    #    col_names=["input_size", "output_size", "num_params", "trainable"],
    #    col_width=20,
    #    row_settings=["var_names"]
    #)
    outputs = net(input)
    print('net output size:')
    if isinstance(outputs, tuple):
        for i, output in enumerate(outputs): # If multiple output
            if isinstance(output, tuple): # If this model have multiple branch
                ff,logit=output
                print(f'Branch {i}: Shape of id tensor {ff.shape}, shape of score tensor{logit.shape}')
            else:
                ff,logit = outputs[0], outputs[1]
                if i==0:
                    #ff = output    
                    print(f'This model have only one brach: Shape of id tensor {ff.shape}, shape of score tensor{logit.shape}') # If this model have one branch
    else:
        print(outputs.shape)
