import abc

import torch
import torch.nn.functional as F


class BaseTripletMiner(abc.ABC):
    """ Base class for Triplet Sampler"""

    def __init__(self, cfg):
        self.normalize_embeddings = cfg.LOSSES.TRIPLET_MINING.NORMALIZE
        self.margin = cfg.LOSSES.TRIPLET_MINING.MARGIN

    def __call__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        dist_mat = self.calculate_dist_mat(embeddings)

        dist_ap, dist_an = self.sampling(dist_mat, labels)
        return dist_ap, dist_an

    @staticmethod
    def calculate_dist_mat(embeddings):
        return torch.cdist(embeddings, embeddings)

    @staticmethod
    def get_pos_neg_mat(dist_mat: torch.Tensor, labels: torch.Tensor):
        assert len(dist_mat.size()) == 2
        assert dist_mat.size(0) == dist_mat.size(1)
        N = dist_mat.size(0)

        # matrix which show positive and negative example for each example
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

        dist_mat_pos = dist_mat[is_pos].contiguous().view(N, -1)
        dist_mat_neg = dist_mat[is_neg].contiguous().view(N, -1)
        return dist_mat_pos, dist_mat_neg

    @abc.abstractmethod
    def sampling(self, dist_mat: torch.Tensor, labels: torch.Tensor):
        pass

    def filter(self, ap_mat: torch.Tensor, an_mat: torch.Tensor, condition: bool = None):
        """
        Filter triplet based on margin
        """
        if self.margin > 0.0:
            triplet_margin = ap_mat - an_mat
            threshold_condition = triplet_margin > -self.margin
            if condition is not None:
                threshold_condition &= condition
            ap_mat, an_mat = ap_mat[threshold_condition], an_mat[threshold_condition]
        return ap_mat, an_mat
