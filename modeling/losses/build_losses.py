from modeling.losses.registry import LOSSES
from .triplet_loss import TripletLoss


def build_losses(cfg):
    try:
        return LOSSES.get(cfg.LOSSES.NAME)(cfg)
    except KeyError:
        raise KeyError(f"Loss {cfg.LOSSES.NAME} is not registered in registry")