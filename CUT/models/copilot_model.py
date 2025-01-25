import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from torchvision import models
from .patchnce import PatchNCELoss
import util.util as util
import models.pspnet as pspnet
from torch import nn
import torch.nn.functional as F
from pytorch_metric_learning import losses
from torch.nn import init
from UDAsbs.loss import SoftTripletLoss_vallia, CrossEntropyLabelSmooth, SoftTripletLoss, SoftEntropy
class COPILOTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'copilot']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        if opt.direction == 'AtoB':
            self.num_classes = 751
        else:
            self.num_classes = 702
        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D', 'Copilot']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


        # Segmentation model
        #self.parsing = pspnet.PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, n_classes = 6, deep_features_size=512).to(self.device)
        #self.parsing.load_state_dict(torch.load('/home/jun/pretrained_models/six_parsing.pth'))
        
        #self.netCopilot = ResNet18(num_classes = self.num_classes).to(self.device)
        self.netCopilot = ResNet18(num_classes = self.num_classes).to(self.device)
        #self.netCopilot = Metric_Net().to(self.device)
        self.optimizer_Copilot = torch.optim.Adam(self.netCopilot.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
        self.optimizers.append(self.optimizer_Copilot) 
        self.contrastive = losses.ContrastiveLoss()
        #self.criterion_ce = CrossEntropyLabelSmooth(self.num_classes).to(self.device)
        #self.criterion_triple = SoftTripletLoss_vallia(margin=0.0).to(self.device)
    
                      
    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # Compute copilot loss
        self.loss_copilot= self.compute_copilot_loss()
        

        # update D
        self.set_requires_grad(self.netD, True)
        self.set_requires_grad(self.netCopilot, False)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward(retain_graph=True)  # Retain graph if needed for subsequent backward pass
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netCopilot, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss() + 3*self.loss_copilot
        self.loss_G.backward(retain_graph=True)  # Retain graph if needed for subsequent backward pass
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

        # update Resnet
        self.set_requires_grad(self.netCopilot, True)
        self.optimizer_Copilot.zero_grad()
        self.loss_copilot.backward()  # No need to retain the graph here
        self.optimizer_Copilot.step()
         
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.label_A = input['A_id' if AtoB else 'B_id'].to(self.device)
        self.label_B = input['B_id' if AtoB else 'A_id'].to(self.device)
        
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # Caculate a_real segmentation map
        
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]
    
    
    def compute_copilot_loss(self):
        fake = self.fake_B.detach()
        features = self.netCopilot(fake)
        loss_con = self.contrastive(features, self.label_A)
        
        return loss_con
    
    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)


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


# Define the ResNet50-based Model
class ResNet18(nn.Module):
    def __init__(self, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=751):
        super().__init__()
        model_ft = models.resnet18(pretrained=pretrained)
        #model_ft.layer4[0].downsample[0].stride = (1,1)
        #model_ft.layer4[0].conv2.stride = (1,1)
        self.model = model_ft
        
        self.cut_at_pooling = cut_at_pooling
        print("GeneralizedMeanPoolingP")
        self.gap = GeneralizedMeanPoolingP(3)
        
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
    
    def forward(self, x):
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

        if self.norm:#FALSE
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:#FALSE
            bn_x = F.relu(bn_x)
        if self.dropout > 0:#FALSE
            bn_x = self.drop(bn_x)
        
        return bn_x

class Conv_Relu_Pool(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Conv_Relu_Pool, self).__init__()
        self.l = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.l(x)

class Metric_Net(nn.Module):

    def __init__(self, dim=64):
        super(Metric_Net, self).__init__()

        self.ls = nn.Sequential(
            nn.Conv2d(3, dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2),
            Conv_Relu_Pool(dim, dim*2),
            Conv_Relu_Pool(dim*2, dim*4),
        )
		
        self.fc1 = nn.Linear(2048, dim*2, bias=None)
        self.relu1 = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(dim*2, dim, bias=None)
        
    def forward(self, x):
        x = self.ls(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim =1, eps=1e-12)
        return x