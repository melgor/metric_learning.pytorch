import os
from collections import OrderedDict

import torch
from torch.nn.modules import Sequential

from modeling.models.registry import BACKBONES, HEADS
from .backbone.resnet50 import ResNet50
from .head.linear import Linear

#TODO: Current Register does not allow printing list of regiseted obejects.
def build_backbone(cfg):
    try:
        return BACKBONES.get(cfg.MODEL.BACKBONE.NAME)()
    except KeyError:
        raise KeyError(f"backbone {cfg.MODEL.BACKBONE} is not registered in registry")


def build_head(cfg):
    try:
        return HEADS.get(cfg.MODEL.HEAD.NAME)(in_channels=cfg.MODEL.BACKBONE.OUT_FEATURES,
                                              out_channels=cfg.MODEL.HEAD.DIM)
    except KeyError:
        raise KeyError(f"head {cfg.MODEL.NAME} is not registered in registry")


def build_model(cfg):
    # print(cfg)
    backbone = build_backbone(cfg)
    head = build_head(cfg)

    model = Sequential(OrderedDict([
        ('backbone', backbone),
        ('head', head)
    ]))

    if cfg.MODEL.PRETRAIN == 'imagenet':
        print('Loading imagenet pretrianed model ...')
        pretrained_path = os.path.expanduser(cfg.MODEL.PRETRIANED_PATH[cfg.MODEL.BACKBONE.NAME])
        model.backbone.load_param(pretrained_path)
    elif os.path.exists(cfg.MODEL.PRETRAIN):
        ckp = torch.load(cfg.MODEL.PRETRAIN)
        model.load_state_dict(ckp['model'])

    model = torch.nn.DataParallel(model)
    return model
