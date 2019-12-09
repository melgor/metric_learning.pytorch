from modeling.miners.registry import MINING
from .triplet_all import TripletAll
from .triplet_hard import TripletHard


def build_mining(cfg):
    try:
        return MINING.get(cfg.LOSSES.MINING)(cfg)
    except KeyError:
        raise KeyError(f"MINING {cfg.LOSSES.MINING} is not registered in registry")