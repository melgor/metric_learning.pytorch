import os
import logging

import pandas as pd
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
import torch.backends.cudnn as cudnn

from data.datasets import CUB200, build_transforms
from engine.trainers import BaseTrainer
from engine.testers import BaseTester
from utils.data_logger import find_device

cudnn.benchmark = True


class Engine:
    def __init__(self,
                 cfg,
                 models,
                 optimizers,
                 lr_schedulers,
                 loss_funcs,
                 mining_funcs=lambda x, y: x,
                 sampler=None
                 ):
        self.epochs = cfg.SOLVER.EPOCHS
        self.dataset_root = cfg.DATA.DATA_ROOT
        self.train_batch_size = cfg.DATA.TRAIN_BATCHSIZE
        self.test_batch_size = cfg.DATA.TEST_BATCHSIZE
        self.num_instances = cfg.DATA.NUM_INSTANCES
        self.dataloader_num_workers = cfg.DATA.NUM_WORKERS
        self.save_dir = cfg.SAVE_DIR

        train_dataset = CUB200(dataset_root=self.dataset_root, transform=build_transforms(cfg, is_train=True), train=True)
        train_sampler = sampler(train_dataset, self.train_batch_size, num_instances=self.num_instances)
        self.trainer = BaseTrainer(models=models, optimizers=optimizers, lr_schedulers=lr_schedulers, loss_funcs=loss_funcs,
                                   train_dataset=train_dataset, batch_size=self.train_batch_size, dataloader_num_workers=self.dataloader_num_workers,
                                   mining_funcs=mining_funcs, sampler=train_sampler)

        self.tester_train = None
        if cfg.VALIDATION.TEST_TRAIN:
            train_test_dataset = CUB200(dataset_root=self.dataset_root, transform=build_transforms(cfg, is_train=False), train=True)
            self.tester_train = BaseTester(cfg=cfg, models=models, test_dataset=train_test_dataset, batch_size=self.test_batch_size,
                                           dataloader_num_workers=self.dataloader_num_workers, name="test_train")

        test_dataset = CUB200(dataset_root=self.dataset_root, transform=build_transforms(cfg, is_train=False), train=False)
        self.tester_test = BaseTester(cfg=cfg, models=models, test_dataset=test_dataset, batch_size=self.test_batch_size,
                                      dataloader_num_workers=self.dataloader_num_workers, name="test_test")
        self.logger = logging.getLogger(cfg.LOGGER.NAME)
        self.models = models.to(find_device())

        checkpoint_dir = os.path.join(self.save_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpointer = Checkpointer(models, checkpoint_dir, optimizer=optimizers)
        self.periodic_checkpointer = PeriodicCheckpointer(checkpointer=self.checkpointer, period=1, max_iter=10e6)

    def run(self):
        logs_buffer = []
        for epoch in range(self.epochs):
            self.logger.info(f"Start Epoch: {epoch}")
            epoch_logs = {"epoch": epoch}
            train_logs = self.trainer.train()
            epoch_logs.update(train_logs)

            if self.tester_train is not None:
                self.logger.info(f"Test Train-Dataset Epoch: {epoch}")
                accuracy = self.tester_train.test()
                epoch_logs.update(accuracy)

            self.logger.info(f"Test Test-Dataset Epoch: {epoch}")
            accuracy = self.tester_test.test()
            epoch_logs.update(accuracy)

            self.periodic_checkpointer.step(epoch, **epoch_logs)
            logs_buffer.append(epoch_logs)

            # save all logs as CSV file at every epoch
            data_logs = pd.DataFrame(logs_buffer)
            data_logs.to_csv(f"{self.save_dir}/train_logs.csv")




