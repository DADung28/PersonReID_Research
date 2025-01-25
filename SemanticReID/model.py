import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.nn import functional as F
from torch.autograd import Variable
from net.pspnet import PSPNet
import pretrainedmodels
import timm
from utils.utils import load_state_dict_mute
import os
from torchinfo import summary
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
            
class ResParsing(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, stride=1, test=False, linear_num=512):
        super().__init__()
        
        self.softmax = torch.nn.Softmax(dim=1)
        self.conv1 = nn.Conv2d(in_channels=21, out_channels=3, kernel_size=1, stride=1, padding=0)
        PSP = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 6, deep_features_size=512, backend='densenet')
        PSP.load_state_dict(torch.load('./parsed_model/six_parsing.pth'))
        self.PSP = PSP
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=class_num)
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, test = test)
    def forward(self, x):
        with torch.inference_mode():
            preds, _= self.PSP(x) # torch.Size([32, 6, 224, 224])
        
        p = preds.clone().requires_grad_(True)
        filters = p[:,1:,:,:]
        # Pass preds through softmax layer
        preds = self.softmax(preds) # (16, 6, 224, 224)
        # Get all the filter of head, body, arm, leg and feat
        filters = preds[:,1:,:,:]
        # Get the foreground filter by sum up all the filter of head, body, arm, leg and feat
        fg = torch.sum(filters, dim = 1).unsqueeze(1)
        # Pack all foregound, head, body, arm, leg and feat into one
        filter_all = torch.cat((fg, filters), dim = 1) # torch.Size([32, 6, 224, 224])

        expanded_filter_all = filter_all.unsqueeze(2) # torch.Size([5, 6, 1, 224, 224])
        expanded_x = x.unsqueeze(1) # torch.Size([5, 1, 3, 224, 224]) 
        # Pass image though 6 filter by element-wise multiply
        x_parsing = torch.mul(expanded_filter_all, expanded_x) # torch.Size([5, 6, 3, 224, 224])
        # Pack the original image the input
        x = torch.cat((x.unsqueeze(1), x_parsing), dim = 1) # torch.Size([5, 7, 3, 224, 224])
        x = x.view(x.shape[0],-1,x.shape[3],x.shape[4])

        x = self.conv1(x)
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


# Define the ViT Model with cls token
class ViT(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, stride=1, test=False, linear_num=512, size=384):
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
        x = self.avgpool(x) # torch.Size([32, 197, 768]) 
        x = self.dropout(x)
        x = x.view(x.shape[0], x.shape[2])
        x = self.classifier(x)
        return x

# Define the ViT Model with cls token
class ViT_Backbone(nn.Module):
    def __init__(self, size=384):
        super().__init__()
        #model = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=1)
        model = timm.create_model(f'vit_base_patch16_{size}', pretrained=True, num_classes=1)
        #model = timm.create_model('vit_huge_patch16_gap_448', pretrained=True, num_classes=1)
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
        self.head = nn.Sequential()
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
        return x

# Define the ViT Model with cls token
class PIViT(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, test=False, linear_num=512,size=384):
        super().__init__()
        self.test = test
        #self.conv1 = nn.Conv2d(in_channels=21, out_channels=3, kernel_size=1, stride=1, padding=0)
        parsing_model = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 2, deep_features_size=512)
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/binary_parsing.pth'))
        self.parsing_model = parsing_model
        self.model = ViT_Backbone(size)
        self.classifier = ClassBlock(768, class_num, droprate, linear=linear_num, test= test)
    def forward(self, x):
        x = F.interpolate(input=x, size=(x.shape[2], x.shape[3]//2), mode='bilinear', align_corners=False) 
        with torch.inference_mode():
            p , class_score= self.parsing_model(x)

        p = p.clone().requires_grad_(True)
        # Pass get 0,1 map
        p = torch.argmax(p, dim=1).unsqueeze(1) # torch.Size([16, 2, 28, 28])
        # Get all the filter of head, body, arm, leg and feat
    
        # Get the foreground filter by sum up all the filter of head, body, arm, leg and feat
        x = torch.cat((x,x * p), dim = 3)
        
        x = self.model(x)
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
        #x = F.interpolate(input=x, size=(x.shape[2], x.shape[3]//2), mode='bilinear', align_corners=False) 
        with torch.inference_mode():
            p , class_score= self.parsing_model(x)

        p = p.clone().requires_grad_(True)
        # Pass get 0,1 map
        p = torch.argmax(p, dim=1).unsqueeze(1) # torch.Size([16, 2, 28, 28])
        # Get all the filter of head, body, arm, leg and feat
    
        # Get the foreground filter by sum up all the filter of head, body, arm, leg and feat
        x = torch.cat((x,x * p), dim = 3)
        
        x = self.model(x)
        x = self.classifier(x)
        return x

class PIResNet50_small(nn.Module):
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
        x = F.interpolate(input=x, size=(x.shape[2], x.shape[3]//2), mode='bilinear', 
                          align_corners=False) 
        with torch.inference_mode():
            p , class_score= self.parsing_model(x)

        p = p.clone().requires_grad_(True)
        # Pass get 0,1 map
        p = torch.argmax(p, dim=1).unsqueeze(1) # torch.Size([16, 2, 28, 28])
        # Get all the filter of head, body, arm, leg and feat
    
        # Get the foreground filter by sum up all the filter of head, body, arm, leg and feat
        x = torch.cat((x,x * p), dim = 3)
        
        x = self.model(x)
        x = self.classifier(x)
        return x

# Define the ViT Model with cls token
class PIViT_old(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, test=False, linear_num=512,size=384):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.test = test
        #self.conv1 = nn.Conv2d(in_channels=21, out_channels=3, kernel_size=1, stride=1, padding=0)
        parsing_model = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 6, deep_features_size=512)
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/six_parsing.pth'))
        self.parsing_model = parsing_model
        self.model = ViT_Backbone(size)
        self.classifier = ClassBlock(768, class_num, droprate, linear=linear_num, test= test)
    def forward(self, x):
        x_0 = x
        with torch.inference_mode():
            p , class_score= self.parsing_model(x_0)
            p = F.interpolate(input=p, size=(x_0.shape[2], x_0.shape[3]), mode='bilinear', align_corners=False) # (16,6,56,56)

        x_parsing = p.clone().requires_grad_(True)
        # Pass preds through softmax layer
        x_parsing = self.softmax(x_parsing) # torch.Size([16, 5, 28, 28])
        # Get all the filter of head, body, arm, leg and feat
        filters = x_parsing[:,1:,:,:]
        # Get the foreground filter by sum up all the filter of head, body, arm, leg and feat
        fg = torch.sum(filters, dim = 1).unsqueeze(1) # torch.Size([32, 1, 224, 224])
        x = torch.cat((x,x*fg), dim = 3)
        
        x = self.model(x)
        x = self.classifier(x)
        return x

# Define the ViT Model with cls token
class PIViT_nohead(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, test=False, linear_num=512,size=384):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.test = test
        #self.conv1 = nn.Conv2d(in_channels=21, out_channels=3, kernel_size=1, stride=1, padding=0)
        parsing_model = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 6, deep_features_size=512)
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/six_parsing.pth'))
        self.parsing_model = parsing_model
        self.model = ViT_Backbone(size)
        self.classifier = ClassBlock(768, class_num, droprate, linear=linear_num, test= test)
    def forward(self, x):
        
        with torch.inference_mode():
            p , class_score= self.parsing_model(x)
            p = F.interpolate(input=p, size=(x.shape[2], x.shape[3]//2), mode='bilinear', align_corners=False) # (16,6,56,56)

        x = F.interpolate(input=x, size=(x.shape[2], x.shape[3]//2), mode='bilinear', align_corners=False)
        x_parsing = p.clone().requires_grad_(True)
        # Pass preds through softmax layer
        x_parsing = self.softmax(x_parsing) # torch.Size([16, 5, 28, 28])
        # Get all the filter of head, body, arm, leg and feat
        #filters = x_parsing[:,1:,:,:]
        filters_nohead = x_parsing[:,2:,:,:]
        
        # Get the foreground filter by sum up all the filter of head, body, arm, leg and feat
        #fg = torch.sum(filters, dim = 1).unsqueeze(1) # torch.Size([32, 1, 224, 224])
        nohead = torch.sum(filters_nohead, dim = 1).unsqueeze(1)
        x = torch.cat((x,x*nohead), dim = 3)

        x = self.model(x)
        x = self.classifier(x)
        return x
 
# Define the ViT Model with cls token
class PIViT_3part(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, test=False, linear_num=512, size=384):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.test = test
        #self.conv1 = nn.Conv2d(in_channels=21, out_channels=3, kernel_size=1, stride=1, padding=0)
        parsing_model = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 6, deep_features_size=512)
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/six_parsing.pth'))
        self.parsing_model = parsing_model
        self.model = ViT_Backbone(size)
        self.classifier = ClassBlock(768, class_num, droprate, linear=linear_num, test= test)
    def forward(self, x):
        
        with torch.inference_mode():
            p , class_score= self.parsing_model(x)
            p = F.interpolate(input=p, size=(x.shape[2], x.shape[3]//3), mode='bilinear', align_corners=False) # (16,6,56,56)

        x = F.interpolate(input=x, size=(x.shape[2], x.shape[3]//3), mode='bilinear', align_corners=False)
        x_parsing = p.clone().requires_grad_(True)
        # Pass preds through softmax layer
        x_parsing = self.softmax(x_parsing) # torch.Size([16, 5, 28, 28])
        # Get all the filter of head, body, arm, leg and feat
        filters = x_parsing[:,1:,:,:]
        filters_nohead = x_parsing[:,2:,:,:]
        # Get the foreground filter by sum up all the filter of head, body, arm, leg and feat
        fg = torch.sum(filters, dim = 1).unsqueeze(1) # torch.Size([32, 1, 224, 224])
        nohead = torch.sum(filters_nohead, dim = 1).unsqueeze(1)
        x = torch.cat((x,x*fg,x*nohead), dim = 3)
        
        x = self.model(x)
        x = self.classifier(x)
        return x
    
# Define the ViT Model with cls token
class PIViT_4part(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, test=False, linear_num=512, size=384):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.test = test
        #self.conv1 = nn.Conv2d(in_channels=21, out_channels=3, kernel_size=1, stride=1, padding=0)
        parsing_model = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 4, deep_features_size=512)
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/four_parsing.pth', map_location=torch.device('cuda')))
        self.parsing_model = parsing_model
        self.model = ViT_Backbone(size)
        self.classifier = ClassBlock(768, class_num, droprate, linear=linear_num, test= test)
    def forward(self, x):
        x = F.interpolate(input=x, size=(x.shape[2]//2, x.shape[3]//2), mode='bilinear', align_corners=False) # (16,6,,56)
        with torch.inference_mode():
            p , class_score= self.parsing_model(x)

        x_parsing = p.clone().requires_grad_(True)
        # Pass preds through softmax layer
        x_parsing = self.softmax(x_parsing) # torch.Size([16, 5, 28, 28])
        # Get all the filter of head, body, arm, leg and feat
        x_foreground = 1 - x_parsing[:,0:1,:,:]# torch.Size([16, 1, 28, 28]) 
        # Replace background propability with foregroiund probability
        x_parsing = torch.cat((x_foreground, x_parsing[:,1:,:,:]), dim=1)
        # Get the foreground filter by sum up all the filter of head, body, arm, leg and feat
        x_parsing = F.normalize(x_parsing, p=1, dim=1)
        foreground = x * x_parsing[:,0:1,:,:]
        body = x * (torch.sum(x_parsing[:,1:3,:,:], dim=1).unsqueeze(1))
        leg = x * x_parsing[:,3:4,:,:]
        #print(x.shape, foreground.shape, body.shape, leg.shape)
        x_up = torch.cat((x,foreground), dim = 3)
        x_down = torch.cat((body, leg), dim=3)
        x = torch.cat((x_up, x_down), dim=2)
        x = self.model(x)
        x = self.classifier(x)
        return x
    



# Define the ViT Model with cls token
class TwinViT(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, test=False, linear_num=512, size=384):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.test = test
        parsing_model = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 2, deep_features_size=512)
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/binary_parsing.pth', map_location=torch.device('cuda')))
        self.parsing_model = parsing_model
        self.head = 2
        self.model = ViT_Backbone(size)
        self.classifier0 = ClassBlock(768, class_num, droprate, linear=linear_num, test= test)
        self.classifier1 = ClassBlock(768, class_num, droprate, linear=linear_num, test= test)
        
    def forward(self, x):
        with torch.inference_mode():
            p , class_score= self.parsing_model(x)
            p = F.interpolate(input=p, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False) # (16,6,56,56)
        x_parsing = p.clone().requires_grad_(True)
        # Get all the filter of head, body, arm, leg and feat
        x_parsing = self.softmax(x_parsing) # torch.Size([16, 5, 28, 28])
        # Get foreground probabilites by do 1-background probabiliry 
        x_parsing = F.normalize(x_parsing, p=1, dim=1)
        inputs = []
        inputs.append(x)
        x_fg = x*x_parsing[:,1:2,:,:]
        inputs.append(x_fg)
        outputs = []
        
        
        for i in range(self.head):
            classifier = getattr(self,'classifier'+str(i))
            model = self.model
            output = classifier(model(inputs[i]))
            outputs.append(output)       
        if self.test:
            return torch.cat((outputs), dim=1) 
        else:
            return outputs


class ViTSemantic(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, stride=1, test=False, linear_num=512):
        super().__init__()
        PSP = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet')
        PSP.load_state_dict(torch.load('parsing_all.pth'))
        self.PSP = PSP
        self.test = test
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
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1,768))
        self.dropout = nn.Dropout(p=0.5)
        self.part = 20
        self.classifier0 = ClassBlock(768, class_num, droprate, linear=linear_num, test=self.test) 
        for i in range(self.part):
            name = 'classifier'+str(i+1)
            setattr(self, name, ClassBlock(768, class_num, droprate, linear=linear_num, test=self.test))
    def forward(self, x):
        with torch.inference_mode():
            f, _= self.PSP(x) 
            f = F.interpolate(input=f, size=(14, 14), mode='bilinear', align_corners=False) # (16,2,14,14)
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) 
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x) # torch.Size([32, 197, 768])
        x = self.norm(x) # torch.Size([32, 197, 768])
        x = self.fc_norm(x) # torch.Size([32, 197, 768]) 
        #x = self.avgpool(x) # torch.Size([32, 197, 768]) 
        x = self.dropout(x)
        x_global = self.avgpool2(x)  # torch.Size([32, 768])
        x_global = self.classifier0(x_global)
        x_local = x[:,1:,:].view(-1,768,14,14) #  torch.Size([32, 768, 14, 14])
        segmap = f.clone().requires_grad_(True)
        
        #x = torch.matmul(x, f_copy) # (16, 2048, 20)
        part = {}
        y_local = []
        y_local.append(x_global)
        for i in range(self.part):
            mask = segmap[:,i,:].unsqueeze(1)
            part[i] = self.avgpool(torch.mul(x_local, mask)).squeeze() # (16, 768)
            name = 'classifier'+str(i+1)
            c = getattr(self,name)
            y_local.append(c(part[i]))
            
        
        if self.test:
            features = y_local[0]
            
            for output in y_local[1:]:
                features = torch.cat((features,output), dim=1) # (16, 2048*20)
            return features
        else:
            #print(len(y_local))
            return y_local

class Swin_Backbone(nn.Module):
    def __init__(self, size=384):
        super().__init__()
        #model_ft = timm.create_model('swin_base_patch4_window7_224', pretrained=True, drop_path_rate = 0.2)
        if size == 384:
            model_ft = timm.create_model('swin_base_patch4_window12_384', pretrained=True, drop_path_rate = 0.2)
        else:
            model_ft = timm.create_model('swin_base_patch4_window7_224', pretrained=True, drop_path_rate = 0.2)
        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.head = nn.Sequential() # save memory
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        x = self.model.forward_features(x)
        x = x.view(x.shape[0],-1,x.shape[-1]) # Change shape from [batchsize, 7, 7, 1024] -> [batchsize, 49, 1024]
        # swin is update in latest timm>0.6.0, so I add the following two lines.
        x = self.avgpool(x.permute((0,2,1))).squeeze() # Change shape from [batchsize, 49, 1024] -> [batchsize, 1024, 49] and do average pooling on 2 dimension
        return x 

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
class PISwin(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, test=False, linear_num=512, size=384):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.test = test
        #self.conv1 = nn.Conv2d(in_channels=21, out_channels=3, kernel_size=1, stride=1, padding=0)
        parsing_model = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 2, deep_features_size=512)
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/binary_parsing.pth'))
        self.parsing_model = parsing_model
        self.model = Swin_Backbone(size=size)
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, test= test)
    def forward(self, x):
        with torch.inference_mode():
            p , class_score= self.parsing_model(x)
            p = F.interpolate(input=p, size=(x.shape[2], x.shape[3]//2), mode='bilinear', align_corners=False) # (16,6,56,56)
        x = F.interpolate(input=x, size=(x.shape[2], x.shape[3]//2), mode='bilinear', align_corners=False) # (16,6,,56)
        x_parsing = p.clone().requires_grad_(True)
        # Pass preds through softmax layer
        x_parsing = self.softmax(x_parsing) # torch.Size([16, 5, 28, 28])
        # Get all the filter of head, body, arm, leg and feat
        fg = x_parsing[:,1:2,:,:]# torch.Size([32, 1, 224, 224])
        
        x = torch.cat((x,x*fg), dim = 3)
        x = self.model(x)
        x = self.classifier(x)
        return x
    
# Define the ResNet50-based Model
class ResNet50(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, stride=2, test=False, linear_num=512):
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
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x
    
# Define the ViT Model with cls token
class PIResNet50_old(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, stride=2, test=False, linear_num=512):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.test = test
        #self.conv1 = nn.Conv2d(in_channels=21, out_channels=3, kernel_size=1, stride=1, padding=0)
        parsing_model = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 2, deep_features_size=512)
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/binary_parsing.pth'))
        self.parsing_model = parsing_model
        self.model = ResNet50(class_num=751, droprate=0.5, stride=2, test=False, linear_num=512)
        self.model.classifier = nn.Sequential()
        self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, test= test)
    def forward(self, x):
        with torch.inference_mode():
            p , class_score= self.parsing_model(x)
            p = F.interpolate(input=p, size=(x.shape[2], x.shape[3]//2), mode='bilinear', align_corners=False) # (16,6,56,56)
        x = F.interpolate(input=x, size=(x.shape[2], x.shape[3]//2), mode='bilinear', align_corners=False) # (16,6,,56)
        x_parsing = p.clone().requires_grad_(True)
        # Pass preds through softmax layer
        x_parsing = self.softmax(x_parsing) # torch.Size([16, 5, 28, 28])
        # Get all the filter of head, body, arm, leg and feat
        fg = x_parsing[:,1:2,:,:]# torch.Size([32, 1, 224, 224])
        x = torch.cat((x,x*fg), dim = 3)
        x = self.model(x)
        x = self.classifier(x)
        return x

class TwinResNet50(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, stride=1, test=False, linear_num=512):
        super().__init__()
        parsing_model = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 2, deep_features_size=512)
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/binary_parsing.pth', map_location=torch.device('cuda')))
        self.parsing_model = parsing_model
        self.softmax = torch.nn.Softmax(dim=1)
        self.head = 2
        self.test = test
        self.model = ResNet50(class_num, droprate, stride, test, linear_num)
        self.model.classifier = nn.Sequential()
        self.classifier0 = ClassBlock(2048, class_num, droprate, linear=linear_num, test= test)
        self.classifier1 = ClassBlock(2048, class_num, droprate, linear=linear_num, test= test)
        #for i in range(self.head):
        #    name = 'model'+str(i)
        #    setattr(self, name, ResNet50(class_num, droprate, stride, test, linear_num))
            
    def forward(self, x):
        with torch.inference_mode():
            p , class_score= self.parsing_model(x)
            p = F.interpolate(input=p, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False) # (16,6,56,56)
        x_parsing = p.clone().requires_grad_(True)
        # Get all the filter of head, body, arm, leg and feat
        x_parsing = self.softmax(x_parsing) # torch.Size([16, 5, 28, 28])
        # Get foreground probabilites by do 1-background probabiliry 
        x_parsing = F.normalize(x_parsing, p=1, dim=1)
        inputs = []
        inputs.append(x)
        x_fg = x*x_parsing[:,1:2,:,:]
        inputs.append(x_fg)
        outputs = []
        
        
        for i in range(self.head):
            classifier = getattr(self,'classifier'+str(i))
            model = self.model
            output = classifier(model(inputs[i]))
            outputs.append(output)       
        if self.test:
            return torch.cat((outputs), dim=1) 
        else:
            return outputs

class ResNet50_4head(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, stride=1, test=False, linear_num=512):
        super().__init__()
        parsing_model = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 4, deep_features_size=512)
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/four_parsing.pth', map_location=torch.device('cuda')))
        self.parsing_model = parsing_model
        self.softmax = torch.nn.Softmax(dim=1)
        self.head = 4
        self.test = test
        self.model = ResNet50(class_num, droprate, stride, test, linear_num)
        self.model.classifier = nn.Sequential()
        for i in range(self.head):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, class_num, droprate, linear=linear_num, test= test))
            
    def forward(self, x):
        with torch.inference_mode():
            p , class_score= self.parsing_model(x)
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
        inputs = []
        inputs.append(x)
        
        for i in range(x_parsing.shape[1]):
            mask = x_parsing[:,i:i+1,:]
            x_part = torch.mul(x, mask)# (16, 2048)
            inputs.append(x_part)

        outputs = []
        for i in range(self.head):
            classifier = getattr(self,'classifier'+str(i))
            model = self.model
            output = classifier(model(inputs[i]))
            outputs.append(output)       
        if self.test:
            return torch.cat((outputs), dim=1) 
        else:
            return outputs
        
class ResNet50_6head(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, stride=1, test=False, linear_num=512):
        super().__init__()
        parsing_model = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 6, deep_features_size=512)
        parsing_model.load_state_dict(torch.load('/home/jun/pretrained_models/six_parsing.pth', map_location=torch.device('cuda')))
        self.parsing_model = parsing_model
        self.softmax = torch.nn.Softmax(dim=1)
        self.head = 4
        self.test = test
        for i in range(self.head):
            name = 'model'+str(i)
            setattr(self, name, ResNet50(class_num, droprate, stride, test, linear_num))
            
    def forward(self, x):
        with torch.inference_mode():
            p , class_score= self.parsing_model(x)
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
        inputs = []
        inputs.append(x)
        
        for i in range(x_parsing.shape[1]):
            mask = x_parsing[:,i:i+1,:]
            x_part = torch.mul(x, mask)# (16, 2048)
            inputs.append(x_part)

        outputs = []
        for i in range(self.head):
            model = getattr(self,'model'+str(i))
            output = model(inputs[i])
            outputs.append(output)       
        
        if self.test:
            return torch.cat((outputs), dim=1) 
        else:
            return outputs

class ResSemantic(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, stride=1, test=False, linear_num=512):
        super().__init__()
        PSP = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet')
        PSP.load_state_dict(torch.load('parsing_all.pth'))
        self.PSP = PSP
        self.test = test
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.part = 20
        self.classifier0 = ClassBlock(2048, class_num, droprate, linear=linear_num, test=self.test) 
        self.classifier1 = ClassBlock(2048, class_num, droprate, linear=linear_num, test=self.test)             
        self.classifier2 = ClassBlock(2048*19, class_num, droprate, linear=linear_num, test=self.test)      
    def forward(self, x):
        with torch.inference_mode():
            f, _= self.PSP(x) 
            f = F.interpolate(input=f, size=(14, 14), mode='bilinear', align_corners=False) # (16,20,14,14)
              
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x_global = self.model.avgpool(x).squeeze() # (16,2048)
        y_global = self.classifier0(x_global)
        segmap = f.clone().requires_grad_(True)
        
        #x = torch.matmul(x, f_copy) # (16, 2048, 20)
        #x_part = torch.mul(x, segmap[:,1,:].unsqueeze(1)) 
        x_fg = torch.zeros_like(x)
        y = []
        y.append(y_global)
        for i in range(1,self.part):
            #if i > 1:
                #x_part = torch.cat((x_part, torch.mul(x, segmap[:,i,:].unsqueeze(1))), dim = 1) # concatinate torch.Size([32, 2048, 14, 14]) 19 time to get part representation
            mask = segmap[:,i,:].unsqueeze(1)
            x_fg += torch.mul(x, mask) # add up torch.Size([32, 2048, 14, 14]) 19 time to get foreground representation
        #x_part = self.model.avgpool(x_part).squeeze() # [32, 38912, 14, 14] -> [32, 38912]
        x_fg = self.model.avgpool(x_fg).squeeze() # [32, 2048, 14, 14] -> [32, 2048]
        y_fg = self.classifier1(x_fg)
        #y_part = self.classifier2(x_part)
        y.append(y_fg)
        #y.append(y_part)
        if self.test:
            
            features = y[0]
            for output in y[1:]:
                features = torch.cat((features,output), dim=1) # (16, 2048*20)
                
            return features
        else:
            return y
        #print(out.shape)
        #print(x_fore.shape, x_local.shape, x_global.shape)
        #x,_ = torch.max(x, dim=2)
        #out = out.view(out.size(0), out.size(1))
        #out = self.classifier(out)

    
class PSPResNet(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, stride=1, test=False, linear_num=512):
        super().__init__()
        PSP = PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet')
        PSP.load_state_dict(torch.load('parsing_all.pth'))
        self.PSP = PSP
        self.test = test
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.part = 20
        self.classifier0 = ClassBlock(2048, class_num, droprate, linear=linear_num, test=self.test) 
        self.classifier1 = ClassBlock(2048, class_num, droprate, linear=linear_num*19, test=self.test)             
        self.classifier2 = ClassBlock(2048*19, class_num, droprate, linear=linear_num, test=self.test)      
    def forward(self, x):
        with torch.inference_mode():
            f, _= self.PSP(x) 
            f = F.interpolate(input=f, size=(14, 14), mode='bilinear', align_corners=False) # (16,20,14,14)
              
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x_global = self.model.avgpool(x).squeeze() # (16,2048)
        y_global = self.classifier0(x_global)
        segmap = f.clone().requires_grad_(True)
        
        #x = torch.matmul(x, f_copy) # (16, 2048, 20)
        x_part = torch.mul(x, segmap[:,1,:].unsqueeze(1)) 
        x_fg = torch.zeros_like(x)
        y = []
        y.append(y_global)
        for i in range(1,self.part):
            if i > 1:
                x_part = torch.cat((x_part, torch.mul(x, segmap[:,i,:].unsqueeze(1))), dim = 1) # concatinate torch.Size([32, 2048, 14, 14]) 19 time to get part representation
            mask = segmap[:,i,:].unsqueeze(1)
            x_fg += torch.mul(x, mask) # add up torch.Size([32, 2048, 14, 14]) 19 time to get foreground representation
        x_part = self.model.avgpool(x_part).squeeze() # [32, 38912, 14, 14] -> [32, 38912]
        x_fg = self.model.avgpool(x_fg).squeeze() # [32, 2048, 14, 14] -> [32, 2048]
        y_fg = self.classifier1(x_fg)
        y_part = self.classifier2(x_part)
        y.append(y_fg)
        y.append(y_part)
        if self.test:
            features = y[0]
            for output in y[1:]:
                features = torch.cat((features,output), dim=1) # (16, 2048*20)
            return features
        else:
            return y
        #print(out.shape)
        #print(x_fore.shape, x_local.shape, x_global.shape)
        #x,_ = torch.max(x, dim=2)
        #out = out.view(out.size(0), out.size(1))
        #out = self.classifier(out)
    

# Define the HRNet18-based Model
class HRNet(nn.Module):
    def __init__(self, class_num, droprate=0.5, training=True, linear_num=512):
        super().__init__()
        model_ft = timm.create_model('hrnet_w32', pretrained=True)
        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.classifier = nn.Sequential() # save memory
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, training=training)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

class SPResNet50(nn.Module):
    def __init__(self, class_num=751, droprate=0.5, last_stride=1, test=False, linear_num=512):
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
        self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, test= test)
    
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
            return y
        else:
            return torch.cat(y, dim=1)


'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    #net = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=1)
    net = PISwin()
    #net = ft_net_swin(751, stride=1)
    #summary(model=net, 
    #    input_size=(32, 3, 384, 384), # make sure this is "input_size", not "input_shape"
    #    # col_names=["input_size"], # uncomment for smaller output
    #    col_names=["input_size", "output_size", "num_params", "trainable"],
    #    col_width=20,
    #   row_settings=["var_names"]
    #)
    net = net.to('cpu')
    input = Variable(torch.FloatTensor(8, 3, 224, 224)).to('cpu')
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