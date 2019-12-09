from data.samplers.registry import SAMPLER
from .identity_sampler import IdentitySampler


def build_sampler(cfg):
    try:
        # NOTE: return class, not object of class.
        # Samplers need Dataset which in for available at this stage
        return SAMPLER.get(cfg.DATA.SAMPLER)
    except KeyError:
        raise KeyError(f"SAMPLER {cfg.DATA.SAMPLER} is not registered in registry")
