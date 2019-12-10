from tqdm.autonotebook import tqdm
import torch
import numpy as np
from sklearn.neighbors import KDTree
import logging


class BaseTester:
    def __init__(self,
                 cfg,
                 models,
                 test_dataset,
                 batch_size,
                 dataloader_num_workers,
                 name=""
                 ):
        self.models = models
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.name = name

        self.dataloader = None
        self.losses = {}
        self.device = torch.device("cuda")
        self.logger = logging.getLogger(cfg.LOGGER.NAME)
        self.setup_dataloader()

    def setup_dataloader(self):
        self.dataloader = torch.utils.data.DataLoader(
                                                    self.test_dataset,
                                                    batch_size=int(self.batch_size),
                                                    num_workers=self.dataloader_num_workers,
                                                    shuffle=False,
                                                    pin_memory=True
                                                    )

    def test(self):
        self.logger.info("Start Testing")
        self.set_to_eval()

        embeddings = self.extract_embeddings()
        accuracies = {}
        for K in [1, 5, 10]:
            acc_k = self.accuracy_at_k(np.array(self.test_dataset.targets), embeddings, K, 200)
            accuracies[f"{self.name} Accuraccy at {K}"] = acc_k
            self.logger.info("accuracy@{} = {}".format(K, acc_k))

        self.logger.info("End Testing")
        return accuracies

    def set_to_eval(self):
        self.models.eval()

    @staticmethod
    def accuracy_at_k(y_true: np.ndarray, embeddings: np.ndarray, K: int, sample: int = None) -> float:
        kdtree = KDTree(embeddings)
        if sample is None:
            sample = len(y_true)
        y_true_sample = y_true[:sample]

        indices_of_neighbours = kdtree.query(embeddings[:sample], k=K + 1, return_distance=False)[:, 1:]

        y_hat = y_true[indices_of_neighbours]

        matching_category_mask = np.expand_dims(np.array(y_true_sample), -1) == y_hat

        matching_cnt = np.sum(matching_category_mask.sum(-1) > 0)
        accuracy = matching_cnt / len(y_true_sample)
        return accuracy

    def extract_embeddings(self):
        embeddings = []
        with tqdm(total=len(self.dataloader)) as pbar:
            with torch.no_grad():
                for idx, (data, labels) in enumerate(self.dataloader):
                    embeddings_batch = self.models(data)
                    embeddings.append(embeddings_batch.cpu().detach().numpy())
                    pbar.update()

        return np.vstack(embeddings)
