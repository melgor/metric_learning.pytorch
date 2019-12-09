import torch

from .base_miner import BaseTripletMiner
from .registry import MINING


@MINING.register()
class TripletAll(BaseTripletMiner):
    """
    Idea of sampling is choosing all possible triplet.
    This idea is taken from 'In Defense of the Triplet Loss for Person Re-Identification'
    """

    @staticmethod
    def create_combination(mat_a: torch.Tensor, mat_b: torch.Tensor):
        """
        Create all possible combination of AP and AN. There are a lot of repeated triplets, but implementation is straighforward
        and computations are really easy. It assume that each class have equal number of examples in batch.
        """
        points_par = [(x, y) for x in range(mat_a.size(1)) for y in range(mat_b.size(1))]
        x_pairs = torch.LongTensor([x for x, y in points_par]).cuda()
        y_pairs = torch.LongTensor([y for x, y in points_par]).cuda()

        return mat_a.index_select(1, x_pairs).view(-1), mat_b.index_select(1, y_pairs).view(-1)

    def sampling(self, dist_mat: torch.Tensor, labels: torch.Tensor):
        dist_ap, dist_an = self.get_pos_neg_mat(dist_mat, labels)
        full_ap, full_an = self.create_combination(dist_ap, dist_an)

        return full_ap, full_an
