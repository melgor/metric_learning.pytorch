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
        loss, logs = self.compute_loss(embeddings, labels)

        assert "loss" in logs.keys(), "Each loss function need to return dict with loss key"
        # In case of Nan loss set it to zero
        # TODO: Rethink two situation. One: no pair and triplet, then zero is fine
        #                              Second: Just model exploiding
        if torch.isnan(loss):
            loss = torch.zeros([], requires_grad=True)
            logs["loss"] = 0

        return loss, logs

