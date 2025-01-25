import torch
print(torch.__version__)
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.nn import functional as F
from torch.nn import Parameter
from torch.autograd import Variable
from torch_geometric.nn import knn
from .net.pspnet import PSPNet
from .hrnet.seg_hrnet import HighResolutionNet, HRNet_2Head
import pretrainedmodels
import timm
import os
from torchinfo import summary
from typing import Optional
from .gem_pooling import GeneralizedMeanPoolingP, GeneralizedMeanPoolingP_ViT 
#from .backbones.resnet import BasicBlock, Bottleneck, ResNet
#from .backbones.resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a

#proxy = 'http://10.0.0.107:3128'
#os.environ['http_proxy'] = proxy 
#os.environ['HTTP_PROXY'] = proxy
#os.environ['https_proxy'] = proxy
#os.environ['HTTPS_PROXY'] = proxy
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
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, linear=512):
        super().__init__()
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
        if self.training:
            f = x
            x = self.classifier(x)
            return [x,f]
        else:
            return x
            

# Define the ViT Model with cls token
class ViT(nn.Module):
    def __init__(self, mb_h=768, with_nl=False,pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=None, sour_class=751):
        super().__init__()
        model = timm.create_model(f'vit_base_patch16_224', pretrained=True, num_classes=sour_class)
        self.patch_embed = model.patch_embed
        self.pos_drop = model.pos_drop
        self.cls_token = model.cls_token
        self.pos_embed = model.pos_embed
        self.patch_drop = model.patch_drop
        self.norm_pre = model.norm_pre
        self.blocks = model.blocks
        self.norm_0 = model.norm
        self.fc_norm = model.fc_norm
        self.dropout_0 = nn.Dropout(p=0.5)
        
        self.cut_at_pooling = cut_at_pooling
        print("GeneralizedMeanPoolingP")
        self.gap = GeneralizedMeanPoolingP_ViT(3)
        self.memorybank_fc = nn.Linear(mb_h, mb_h)
        self.mbn=nn.BatchNorm1d(mb_h)
        init.kaiming_normal_(self.memorybank_fc.weight, mode='fan_out')
        init.constant_(self.memorybank_fc.bias, 0)
        
        self.num_features = num_features
        self.norm = norm
        self.dropout = dropout
        self.has_embedding = num_features > 0
        self.num_classes = num_classes

        out_planes = self.norm_0.normalized_shape[0]
        
        # Append new layers
        if self.has_embedding:
            self.feat = nn.Linear(out_planes, self.num_features)
            self.feat_bn = nn.BatchNorm1d(self.num_features)
            init.kaiming_normal_(self.feat.weight, mode='fan_out')
            init.constant_(self.feat.bias, 0)
        else:
            # Change the num_features to CNN output channels
            self.num_features = out_planes
            self.feat_bn = nn.BatchNorm1d(self.num_features)
        self.feat_bn.bias.requires_grad_(False)
        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)
        if self.num_classes is not None:
            for i,num_cluster in enumerate(self.num_classes):
                exec("self.classifier{}_{} = nn.Linear(self.num_features, {}, bias=False)".format(i,num_cluster,num_cluster))
                exec("init.normal_(self.classifier{}_{}.weight, std=0.001)".format(i,num_cluster)) 
        
    def forward(self, x, feature_withbn=False, training=False):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) 
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x) # torch.Size([32, 197, 768])
        x = self.norm_0(x) # torch.Size([32, 197, 768])
        x = self.fc_norm(x) # torch.Size([32, 197, 768]) 
        x = self.dropout_0(x)
        
        # GeneralizedMeanPoolingP(3)
        x = self.gap(x)
         
        x = x.view(x.size(0), -1) # torch.Size([32, 768])
        if self.cut_at_pooling:return x#FALSE

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))#FALSE
        else:
            bn_x = self.feat_bn(x)#1

        if training is False:
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.norm:#FALSE
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:#FALSE
            bn_x = F.relu(bn_x)

        if self.dropout > 0:#FALSE
            bn_x = self.drop(bn_x)

        prob = []
        if self.num_classes is not None:
            for i,num_cluster in enumerate(self.num_classes):
                exec("prob.append(self.classifier{}_{}(bn_x))".format(i,num_cluster))
        else:
            return x, bn_x

        if feature_withbn:#False
           return bn_x, prob

        mb_x = self.mbn(self.memorybank_fc(bn_x))
        return x, prob, mb_x, None
        
    
class Swin(nn.Module):
    def __init__(self, mb_h=1024, with_nl=False,pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=None, sour_class=751):
        super().__init__()
            
        model_ft = timm.create_model('swin_base_patch4_window7_224', pretrained=True, drop_path_rate = 0.2)
        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        model_ft.head = nn.Sequential() # save memory
        self.model = model_ft
        self.cut_at_pooling = cut_at_pooling
        print("GeneralizedMeanPoolingP")
        self.gap = GeneralizedMeanPoolingP(3)
        self.memorybank_fc = nn.Linear(mb_h, mb_h)
        self.mbn=nn.BatchNorm1d(mb_h)
        init.kaiming_normal_(self.memorybank_fc.weight, mode='fan_out')
        init.constant_(self.memorybank_fc.bias, 0)
        
        self.num_features = num_features
        self.norm = norm
        self.dropout = dropout
        self.has_embedding = num_features > 0
        self.num_classes = num_classes

        out_planes = model_ft.norm.normalized_shape[0]

        # Append new layers
        if self.has_embedding:
            self.feat = nn.Linear(out_planes, self.num_features)
            self.feat_bn = nn.BatchNorm1d(self.num_features)
            init.kaiming_normal_(self.feat.weight, mode='fan_out')
            init.constant_(self.feat.bias, 0)
        else:
            # Change the num_features to CNN output channels
            self.num_features = out_planes
            self.feat_bn = nn.BatchNorm1d(self.num_features)
        self.feat_bn.bias.requires_grad_(False)
        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)
        if self.num_classes is not None:
            for i,num_cluster in enumerate(self.num_classes):
                exec("self.classifier{}_{} = nn.Linear(self.num_features, {}, bias=False)".format(i,num_cluster,num_cluster))
                exec("init.normal_(self.classifier{}_{}.weight, std=0.001)".format(i,num_cluster))
    def forward(self, x, feature_withbn=False, training=False):
        x = self.model.forward_features(x) # [batchsize, 7, 7, 1024]
        
        x = x.permute(0,3,1,2) # Change shape from [batchsize, 7, 7, 1024] -> [batchsize, 49, 1024]
        # swin is update in latest timm>0.6.0, so I add the following two lines.
        
        x = self.gap(x)

        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:return x#FALSE

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))#FALSE
        else:
            bn_x = self.feat_bn(x)#1

        if training is False:
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.norm:#FALSE
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:#FALSE
            bn_x = F.relu(bn_x)

        if self.dropout > 0:#FALSE
            bn_x = self.drop(bn_x)

        prob = []
        if self.num_classes is not None:
            for i,num_cluster in enumerate(self.num_classes):
                exec("prob.append(self.classifier{}_{}(bn_x))".format(i,num_cluster))
        else:
            return x, bn_x

        if feature_withbn:#False
           return bn_x, prob
           
        mb_x = self.mbn(self.memorybank_fc(bn_x))
        return x, prob, mb_x, None

# Define the ResNet50-based Model
class ResNet50(nn.Module):
    def __init__(self, mb_h=2048, with_nl=False,pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=None, sour_class=751):
        super().__init__()
        model_ft = models.resnet50(pretrained=pretrained)
        model_ft.layer4[0].downsample[0].stride = (1,1)
        model_ft.layer4[0].conv2.stride = (1,1)
        self.model = model_ft
        
        self.cut_at_pooling = cut_at_pooling
        print("GeneralizedMeanPoolingP")
        self.gap = GeneralizedMeanPoolingP(3)
        self.memorybank_fc = nn.Linear(2048, mb_h)
        self.mbn=nn.BatchNorm1d(mb_h)
        init.kaiming_normal_(self.memorybank_fc.weight, mode='fan_out')
        init.constant_(self.memorybank_fc.bias, 0)
        
        self.num_features = num_features
        self.norm = norm
        self.dropout = dropout
        self.has_embedding = num_features > 0
        self.num_classes = num_classes

        out_planes = model_ft.fc.in_features

        # Append new layers
        if self.has_embedding:
            self.feat = nn.Linear(out_planes, self.num_features)
            self.feat_bn = nn.BatchNorm1d(self.num_features)
            init.kaiming_normal_(self.feat.weight, mode='fan_out')
            init.constant_(self.feat.bias, 0)
        else:
            # Change the num_features to CNN output channels
            self.num_features = out_planes
            self.feat_bn = nn.BatchNorm1d(self.num_features)
        self.feat_bn.bias.requires_grad_(False)
        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)
        if self.num_classes is not None:
            for i,num_cluster in enumerate(self.num_classes):
                exec("self.classifier{}_{} = nn.Linear(self.num_features, {}, bias=False)".format(i,num_cluster,num_cluster))
                exec("init.normal_(self.classifier{}_{}.weight, std=0.001)".format(i,num_cluster))
    def forward(self, x, feature_withbn=False, training=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = self.gap(x)

        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:return x#FALSE

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))#FALSE
        else:
            bn_x = self.feat_bn(x)#1

        if training is False:
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.norm:#FALSE
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:#FALSE
            bn_x = F.relu(bn_x)

        if self.dropout > 0:#FALSE
            bn_x = self.drop(bn_x)

        prob = []
        if self.num_classes is not None:
            for i,num_cluster in enumerate(self.num_classes):
                exec("prob.append(self.classifier{}_{}(bn_x))".format(i,num_cluster))
        else:
            return x, bn_x

        if feature_withbn:#False
           return bn_x, prob
        mb_x = self.mbn(self.memorybank_fc(bn_x))
        return x, prob, mb_x, None

# Define the ResNet50-based Model
class ResNet50_multi(nn.Module):
    def __init__(self, mb_h=2048, with_nl=False,pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=None, sour_class=751):
        super().__init__()
        model_ft = models.resnet50(pretrained=pretrained)
        model_ft.layer4[0].downsample[0].stride = (1,1)
        model_ft.layer4[0].conv2.stride = (1,1)
        self.model = model_ft
        
        self.cut_at_pooling = cut_at_pooling
        print("GeneralizedMeanPoolingP")
        self.gap = GeneralizedMeanPoolingP(3)
        self.memorybank_fc = nn.Linear(2048, mb_h)
        self.mbn=nn.BatchNorm1d(mb_h)
        init.kaiming_normal_(self.memorybank_fc.weight, mode='fan_out')
        init.constant_(self.memorybank_fc.bias, 0)
        
        self.num_features = num_features
        self.norm = norm
        self.dropout = dropout
        self.has_embedding = num_features > 0
        self.num_classes = num_classes

        out_planes = model_ft.fc.in_features

        # Append new layers
        if self.has_embedding:
            self.feat = nn.Linear(out_planes, self.num_features)
            self.feat_bn = nn.BatchNorm1d(self.num_features)
            init.kaiming_normal_(self.feat.weight, mode='fan_out')
            init.constant_(self.feat.bias, 0)
        else:
            # Change the num_features to CNN output channels
            self.num_features = out_planes
            self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn_3 = nn.BatchNorm1d(1024)

        self.feat_bn.bias.requires_grad_(False)
        self.feat_bn_3.bias.requires_grad_(False)

        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)
        if self.num_classes is not None:
            for i,num_cluster in enumerate(self.num_classes):
                exec("self.classifier{}_{} = nn.Linear(self.num_features, {}, bias=False)".format(i,num_cluster,num_cluster))
                exec("init.normal_(self.classifier{}_{}.weight, std=0.001)".format(i,num_cluster))
            for i,num_cluster in enumerate(self.num_classes):
                exec("self.classifier3_{}_{} = nn.Linear(1024, {}, bias=False)".format(i,num_cluster,num_cluster))
                exec("init.normal_(self.classifier3_{}_{}.weight, std=0.001)".format(i,num_cluster))
                
    def forward(self, x, feature_withbn=False, training=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x3 = self.model.layer3(x)
        x4 = self.model.layer4(x3)
       
        x3 = self.gap(x3) 
        x = self.gap(x4)

        x = x.view(x.size(0), -1)
        x3 = x3.view(x3.size(0), -1)

        bn_x = self.feat_bn(x)
        bn_x3 = self.feat_bn_3(x3)

        if self.dropout > 0:#FALSE
            bn_x = self.drop(bn_x)

        prob = []
        prob_3=[]
        if self.num_classes is not None:
            for i,num_cluster in enumerate(self.num_classes):
                exec("prob.append(self.classifier{}_{}(bn_x))".format(i,num_cluster))
            for i, num_cluster in enumerate(self.num_classes):
                exec("prob_3.append(self.classifier3_{}_{}(bn_x3))".format(i, num_cluster))

        else:
            return x, bn_x

        if feature_withbn:#False
           return bn_x, prob

        mb_x = self.mbn(self.memorybank_fc(bn_x))

        # ml_x = self.classifier_ml(bn_x)

        # prob = [F.linear(F.normalize(bn_x), F.normalize(self.weight))]
        # prob_3 = [F.linear(F.normalize(bn_x3), F.normalize(self.weight3))]
        if training is False:
            bn_x = F.normalize(bn_x)
            return bn_x
        return x, prob, mb_x, None, prob_3, x3


# Define the HRNet18-based Model
class HRNet(nn.Module):
    def __init__(self, mb_h=2048, with_nl=False,pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=None, sour_class=751):
        super().__init__()
        model_ft = timm.create_model('hrnet_w32', pretrained=True)
        self.model = model_ft
        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.cut_at_pooling = cut_at_pooling
        print("GeneralizedMeanPoolingP")
        self.gap = GeneralizedMeanPoolingP(3)
        self.memorybank_fc = nn.Linear(2048, mb_h)
        self.mbn=nn.BatchNorm1d(mb_h)
        init.kaiming_normal_(self.memorybank_fc.weight, mode='fan_out')
        init.constant_(self.memorybank_fc.bias, 0)
        
        self.num_features = num_features
        self.norm = norm
        self.dropout = dropout
        self.has_embedding = num_features > 0
        self.num_classes = num_classes

        out_planes = model_ft.classifier.in_features

        # Append new layers
        if self.has_embedding:
            self.feat = nn.Linear(out_planes, self.num_features)
            self.feat_bn = nn.BatchNorm1d(self.num_features)
            init.kaiming_normal_(self.feat.weight, mode='fan_out')
            init.constant_(self.feat.bias, 0)
        else:
            # Change the num_features to CNN output channels
            self.num_features = out_planes
            self.feat_bn = nn.BatchNorm1d(self.num_features)
        self.feat_bn.bias.requires_grad_(False)
        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)
        if self.num_classes is not None:
            for i,num_cluster in enumerate(self.num_classes):
                exec("self.classifier{}_{} = nn.Linear(self.num_features, {}, bias=False)".format(i,num_cluster,num_cluster))
                exec("init.normal_(self.classifier{}_{}.weight, std=0.001)".format(i,num_cluster))

    def forward(self, x, feature_withbn=False, training=False):
        x = self.model.forward_features(x)
        x = self.gap(x)

        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:return x#FALSE

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))#FALSE
        else:
            bn_x = self.feat_bn(x)#1

        if training is False:
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.norm:#FALSE
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:#FALSE
            bn_x = F.relu(bn_x)

        if self.dropout > 0:#FALSE
            bn_x = self.drop(bn_x)

        prob = []
        if self.num_classes is not None:
            for i,num_cluster in enumerate(self.num_classes):
                #print("prob.append(self.classifier{}_{}(bn_x))".format(i,num_cluster))
                exec("prob.append(self.classifier{}_{}(bn_x))".format(i,num_cluster))
        else:
            return x, bn_x

        if feature_withbn:#False
            return bn_x, prob
        
        mb_x = self.mbn(self.memorybank_fc(bn_x))
        return x, prob, mb_x, None


# Define the ResNet50-based Model
class ResNet50_Grouping(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, stride=1, mb_h=512, num_split=2, cluster=False):
        super().__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.cluster = cluster
        self.num_split = num_split
        if self.cluster:
            #self.assignment = KmeanClusterAssignment(cluster_number=32, embedding_dimension=2048)
            self.assignment = ClusterAssignment(cluster_number=32, embedding_dimension=2048)
        self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num)
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        if self.num_split > 1:
            h = x.size(2)
            x1 = []
            xx = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)
            x1.append(xx)
            x1_split = [x[:, :, h // self.num_split * s: h // self.num_split * (s+1), :] for s in range(self.num_split)]
            for xx in x1_split:
                xx = F.avg_pool2d(xx, xx.size()[2:])
                x1.append(xx.view(xx.size(0), -1))   
        else:
            x1 = F.avg_pool2d(x, x.size()[2:])
            x1 = x1.view(x1.size(0), -1)
        #for x_1_test in x1:
        #    print(x_1_test.shape)     
        
        x2 = x2 = F.avg_pool2d(x, x.size()[2:])
        x2 = x2.view(x2.size(0), -1)
        x2 = self.classifier(x2)
        
        if self.training:
            if self.cluster:
                if isinstance(x1, list):
                    x3 = [self.assignment(x) for i, x in enumerate(x1)]
                    #x3 = self.assignment(torch.cat(x1, dim=1))
                    #print(x3)
                    
                else:
                    x3 = self.assignment(x1)
                return [x1, x2, x3]
            else:
                return [x1, x2]
        else:
            x1 = torch.cat(x1, dim=1)
            return [x1, x2]




'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    #net = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=1)
    net = ResNet50_Grouping(cluster=True)
    #net = ft_net_swin(751, stride=1)
    #summary(model=net, 
    #    input_size=(32, 3, 384, 384), # make sure this is "input_size", not "input_shape"
    #    # col_names=["input_size"], # uncomment for smaller output
    #    col_names=["input_size", "output_size", "num_params", "trainable"],
    #    col_width=20,
    #   row_settings=["var_names"]
    #)
    # Sample data
    net = net.to('cpu')
    input = Variable(torch.FloatTensor(32, 3, 384, 384)).to('cpu')
    outputs = net(input)
    print(len(outputs[0]),len(outputs[1]),len(outputs[2]))
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
        pass
        #print(outputs.shape)
