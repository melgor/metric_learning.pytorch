from modeling.miners.registry import MINING
from .triplet_all import TripletAll
from .triplet_hardest import TripletHardest
from .triplet_hard import TripletHard
from .triplet_semi_hard import TripletSemiHard


def build_mining(cfg):
    try:
        return MINING.get(cfg.LOSSES.MINING)(cfg)
    except KeyError:
        raise KeyError(f"MINING {cfg.LOSSES.MINING} is not registered in registry")