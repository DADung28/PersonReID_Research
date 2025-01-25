# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline
from .model import ResNet50, HRNet18, HRNet48, ResNet50_ClassBlock, SPResNet50, SPResNet50_21_loss,SPResNet50_all_part, SPResNet50_all_part_except_global, SPResNet50_foreground_part_global_max, SPResNet50_foreground_part_global_sum, SPResNet50_foreground_part_global_mean
from .SemanticReID import SPResNet50_IMPROVED

def build_model(cfg, num_classes):
    if cfg.MODEL.NAME == 'resnet50':
    #     model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
        model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    elif cfg.MODEL.NAME == 'myresnet50':
        model = ResNet50(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    elif cfg.MODEL.NAME == 'hrnet18':
        model = HRNet18(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    elif cfg.MODEL.NAME == 'hrnet48':
        model = HRNet48(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    elif cfg.MODEL.NAME == 'resnet50_classblock':
        model = ResNet50_ClassBlock(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    elif cfg.MODEL.NAME == 'SPResNet50':
        model = SPResNet50(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    elif cfg.MODEL.NAME == 'SPResNet50_all_part':
        model = SPResNet50_all_part(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    elif cfg.MODEL.NAME == 'SPResNet50_21_loss':
        model = SPResNet50_21_loss(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    elif cfg.MODEL.NAME == 'SPResNet50_all_part_except_global':
        model = SPResNet50_all_part_except_global(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    elif cfg.MODEL.NAME == 'SPResNet50_foreground_part_global_max':
        model = SPResNet50_foreground_part_global_max(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    elif cfg.MODEL.NAME == 'SPResNet50_foreground_part_global_mean':
        model = SPResNet50_foreground_part_global_mean(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    elif cfg.MODEL.NAME == 'SPResNet50_foreground_part_global_sum':
        model = SPResNet50_foreground_part_global_sum(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    elif cfg.MODEL.NAME == 'SPResNet50_IMPROVED':
        model = SPResNet50_IMPROVED(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    else:
       pass 
    return model

