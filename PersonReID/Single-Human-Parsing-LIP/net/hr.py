import torch
from torch import nn
from torch.nn import functional as F
from config.default import _C as config
from hrnet.seg_hrnet import HighResolutionNet as HRNetv2
from hrnet.seg_hrnet_ocr import HighResolutionNet as HRNetOCR

config.defrost()
config.merge_from_file('config_yml/seg_hrnet_w48.yaml')
config.freeze()

class SegmentationHRNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        model = HRNetv2(cfg, **kwargs)
        model.init_weights(cfg.MODEL.PRETRAINED)
        self.model = model
        self.interpolate = nn.functional.interpolate
    def forward(self, x):
        x = self.model(x)
        x = self.interpolate(input=x, size=(224, 224), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
        return x

class SegmentationHRNetOCR(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        model = HRNetOCR(cfg, **kwargs)
        model.init_weights(cfg.MODEL.PRETRAINED)
        self.model = model
        self.interpolate = nn.functional.interpolate
    def forward(self, x):
        x = self.model(x)
        x_0,x_1 = x[0], x[1]
        x_0 = self.interpolate(input=x_0, size=(224, 224), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
        x_1 = self.interpolate(input=x_1, size=(224, 224), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
        return x_0, x_1
