import torch
import torch.nn.functional as F

from .base_loss import BaseMetricLoss
from .registry import LOSSES


@LOSSES.register()
class TripletLoss(BaseMetricLoss):
    """
    Loss based on FaceNet paper: https://arxiv.org/abs/1503.03832
    loss = |AN - POS| - |AN - NEG| + margin
    Interpretation:
    Anchor should be closer to Positive example than to Negative by margin value
    """
    def __init__(self, cfg):
        self.margin = cfg.LOSSES.TRIPLET_LOSS.MARGIN
        self.only_non_zero = cfg.LOSSES.TRIPLET_LOSS.NON_ZERO
        super().__init__(normalize_embeddings=cfg.LOSSES.TRIPLET_LOSS.NORMALIZE)

    def compute_loss(self, embeddings, labels: torch.Tensor):
        ap, an = embeddings
        logs = {"triplet": ap.size(0)}
        loss = F.relu(ap - an + self.margin)
        if self.only_non_zero:
            # take only example with loss > 0
            loss = loss[loss > 0]
            logs["valid_triplet"] = loss.size(0)
        loss = loss.mean()
        logs["loss"] = loss.item()
        return loss, logs
