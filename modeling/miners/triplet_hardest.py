import torch

from .base_miner import BaseTripletMiner
from .registry import MINING


@MINING.register()
class TripletHardest(BaseTripletMiner):
    """
    Idea of sampling is choosing the harderst positive and negative for each anchor.
    This idea is taken from 'In Defense of the Triplet Loss for Person Re-Identification'
    Code based on idea from https://github.com/Cysu/open-reid
    """

    def sampling(self, dist_mat: torch.Tensor, labels: torch.Tensor):
        N = dist_mat.size(0)
        dist_ap, dist_an = self.get_pos_neg_mat(dist_mat, labels)

        # distances of anchor and positive, take example which is the farthest compared to anchor
        dist_ap, _ = torch.max(dist_ap.contiguous().view(N, -1), 1, keepdim=True)
        # distances of anchor and negative, take example which is the closest compared to anchor
        dist_an, _ = torch.min(dist_an.contiguous().view(N, -1), 1, keepdim=True)

        dist_ap = dist_ap.squeeze(1)
        dist_an = dist_an.squeeze(1)

        return dist_ap, dist_an
