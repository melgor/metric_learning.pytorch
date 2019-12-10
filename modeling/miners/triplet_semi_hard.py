import torch

from .triplet_all import TripletAll
from .registry import MINING


@MINING.register()
class TripletSemiHard(TripletAll):
    """
    Idea of sampling is choosing the semi-hardest positive and negative for each anchor.
    It means that positive is closer to anchor that negative, but still not in the distance > margin

    """
    def __init__(self, cfg):
        super(TripletSemiHard, self).__init__(normalize_embeddings=cfg.LOSSES.TRIPLET_HARD_MINING.NORMALIZE,
                                          margin=cfg.LOSSES.TRIPLET_HARD_MINING.MARGIN)

    def sampling(self, dist_mat: torch.Tensor, labels: torch.Tensor):
        dist_ap, dist_an = self.get_pos_neg_mat(dist_mat, labels)
        full_ap, full_an = self.create_combination(dist_ap, dist_an)

        full_ap, full_an = self.filter(full_ap, full_an, condition=full_ap < full_an)

        return full_ap, full_an
