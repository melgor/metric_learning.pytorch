import torch

from .registry import MINING
from .triplet_all import TripletAll


@MINING.register()
class TripletHard(TripletAll):
    """
    Idea of sampling is choosing the hard positive and negative for each anchor.
    It means that negative is closer to anchor that positive
    """

    def sampling(self, dist_mat: torch.Tensor, labels: torch.Tensor):
        dist_ap, dist_an = self.get_pos_neg_mat(dist_mat, labels)
        full_ap, full_an = self.create_combination(dist_ap, dist_an)

        full_ap, full_an = self.filter(full_ap, full_an, condition=full_ap > full_an)

        return full_ap, full_an
