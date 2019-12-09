from fvcore.common.config import CfgNode as CN

MODEL_PATH = {
    'bninception': "~/.torch/models/bn_inception-52deb4733.pth",
    'ResNet50': "~/.torch/models/resnet50-19c8e357.pth",
}

MODEL_PATH = CN(MODEL_PATH)
