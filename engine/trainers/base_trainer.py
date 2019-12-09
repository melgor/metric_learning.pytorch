import logging
from tqdm.autonotebook import tqdm
import torch

from fvcore.common.history_buffer import HistoryBuffer

from utils.data_logger import find_device

class BaseTrainer:
    def __init__(self,
                 models,
                 optimizers,
                 lr_schedulers,
                 loss_funcs,
                 train_dataset,
                 batch_size,
                 dataloader_num_workers,
                 mining_funcs=lambda x, y: x,
                 sampler=None
                 ):

        self.models = models
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.loss_funcs = loss_funcs
        self.mining_funcs = mining_funcs

        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.sampler = sampler

        self.dataloader = None
        self.losses = HistoryBuffer()
        self.device = find_device()
        self.logger = logging.getLogger('metric.learning')

        self.setup_dataloader()
        # self.models = self.models.to(self.device)

    def setup_dataloader(self):
        self.dataloader = torch.utils.data.DataLoader(
                                                    self.train_dataset,
                                                    batch_size=int(self.batch_size),
                                                    sampler=self.sampler,
                                                    drop_last=True,
                                                    num_workers=self.dataloader_num_workers,
                                                    shuffle=self.sampler is None,
                                                    pin_memory=True
                                                )

    def train(self):
        self.set_to_train()
        with tqdm(total=len(self.dataloader)) as pbar:
            for idx, (data, labels) in enumerate(self.dataloader):
                self.forward_and_backward(data, labels)
                pbar.set_postfix(loss=self.losses.latest(), refresh=False)
                pbar.update()
        self.logger.info(f"End Epoch: Loss Mean Value: {self.losses.avg(window_size=len(self.dataloader))}")

    def set_to_train(self):
        self.models.train()

    def forward_and_backward(self, data, labels):
        data = data.to(self.device)
        labels = labels.to(self.device)
        embeddings = self.models(data)
        embeddings = self.mining_funcs(embeddings, labels)

        # triplet sampler and Loss
        loss = self.loss_funcs(embeddings, labels)
        self.losses.update(loss.item())
        self.optimizers.zero_grad()
        loss.backward()
        self.optimizers.step()


