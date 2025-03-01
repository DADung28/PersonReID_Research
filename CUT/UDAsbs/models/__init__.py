from __future__ import absolute_import
from .model import ResNet50, ResNet50_multi, HRNet, Swin, ViT
from .resnet import *
from .idm_module import *
from .resnet_idm import *
from .resnet_ibn import *
from .resnet_ibn_idm import *
# from .resnet_sbs import resnet50_sbs
from .resnet_multi import resnet50_multi,resnet50_multi_sbs
__factory = {
    'hrnet': HRNet,
    'ResNet50': ResNet50,
    'ResNet50_multi': ResNet50_multi,
    'resnet18': resnet18,
    'ViT': ViT,
    'Swin': Swin,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnet_ibn50a': resnet_ibn50a,
    'resnet_ibn101a': resnet_ibn101a,
    'resnet50_idm': resnet50_idm,
    'resnet_ibn50a_idm': resnet_ibn50a_idm,
    'resnet50_sbs': resnet50_sbs,
    'resnet50_multi': resnet50_multi,
    'resnet50_multi_sbs': resnet50_multi_sbs
}


def names():
    return sorted(__factory.keys())


def create(name, mb_h=2048, sour_class=751, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        Model name. Can be one of 'inception', 'resnet18', 'resnet34',
        'resnet50', 'resnet101', and 'resnet152'.
    pretrained : bool, optional
        Only applied for 'resnet*' models. If True, will use ImageNet pretrained
        model. Default: True
    cut_at_pooling : bool, optional
        If True, will cut the model before the last global pooling layer and
        ignore the remaining kwargs. Default: False
    num_features : int, optional
        If positive, will append a Linear layer after the global pooling layer,
        with this number of output units, followed by a BatchNorm layer.
        Otherwise these layers will not be appended. Default: 256 for
        'inception', 0 for 'resnet*'
    norm : bool, optional
        If True, will normalize the feature to be unit L2-norm for each sample.
        Otherwise will append a ReLU layer after the above Linear layer if
        num_features > 0. Default: False
    dropout : float, optional
        If positive, will append a Dropout layer with this dropout rate.
        Default: 0
    num_classes : int, optional
        If positive, will append a Linear layer at the end as the classifier
        with this number of output units. Default: 0
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    print(f'Creating {name} model with {sour_class} source classes')
    if name in ['resnet50_idm', 'resnet_ibn50a_idm','resnet_ibn50a']:
        return __factory[name](*args, **kwargs)
    else:
        return __factory[name](mb_h=mb_h,sour_class=sour_class,*args, **kwargs)
