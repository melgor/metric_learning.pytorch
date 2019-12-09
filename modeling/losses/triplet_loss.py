import torch
import torch.nn.functional as F

from .base_loss import BaseMetricLoss
from .registry import LOSSES

@LOSSES.register()
class TripletLoss(BaseMetricLoss):

    def __init__(self, cfg):
        self.margin = cfg.LOSSES.TRIPLET_LOSS.MARGIN
        self.only_non_zero = cfg.LOSSES.TRIPLET_LOSS.NON_ZERO
        super().__init__(normalize_embeddings=cfg.LOSSES.TRIPLET_LOSS.NORMALIZE)

    def compute_loss(self, embeddings, labels: torch.Tensor):
        ap, an = embeddings
        loss = F.relu(ap - an + self.margin)
        if self.only_non_zero:
            # take only example with loss > 0
            return loss[loss > 0].mean()
        return loss.mean()
