from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
from .resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a


__all__ = ['ResNetIBN', 'resnet_ibn50a', 'resnet_ibn101a']


class ResNetIBN(nn.Module):
    __factory = {
        '50a': resnet50_ibn_a,
        '101a': resnet101_ibn_a
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, mb_h=2048, sour_class=751):
        super(ResNetIBN, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        resnet = ResNetIBN.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.memorybank_fc = nn.Linear(2048, mb_h)
        self.mbn=nn.BatchNorm1d(mb_h)
        init.kaiming_normal_(self.memorybank_fc.weight, mode='fan_out')
        init.constant_(self.memorybank_fc.bias, 0)
        
        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

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
            #if self.num_classes > 0:
            #    self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
            #    init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        if not pretrained:
            self.reset_params()

    def forward(self, x, feature_withbn=False, training=False):
        x = self.base(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

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


    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def resnet_ibn50a(**kwargs):
    return ResNetIBN('50a', **kwargs)


def resnet_ibn101a(**kwargs):
    return ResNetIBN('101a', **kwargs)
