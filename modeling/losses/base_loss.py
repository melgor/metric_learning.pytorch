import torch
import torch.nn.functional as F


class BaseMetricLoss(torch.nn.Module):

    def __init__(self, normalize_embeddings: bool=True):
        self.normalize_embeddings = normalize_embeddings
        super(BaseMetricLoss, self).__init__()

    def compute_loss(self, embeddings: torch.Tensor, labels: torch.Tensor):
        raise NotImplementedError

    def forward(self, embeddings, labels: torch.Tensor):
        """

        :param embeddings: Can be single Tensor (in case of classification loss)
                            or pair of Tensor (in case of Pair or Triplet Loss)
        :param labels: labels for classification or for pair loss
        :return: Tensor
        """
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        loss = self.compute_loss(embeddings, labels)

        # In case of Nan loss set it to zero
        # TODO: Rethink two situation. One: no pair and triplet, then zero is fine
        #                              Second: Just model exploiding
        if torch.isnan(loss):
            if isinstance(embeddings, tuple):
                loss = torch.sum(embeddings[0] * 0)
                print("NaN Loss")
            elif isinstance(embeddings, torch.Tensor):
                loss = torch.sum(embeddings * 0)
            else:
                raise TypeError(f"Handling zero loss not implemented for embeddings of type {type(embeddings)}")

        return loss

