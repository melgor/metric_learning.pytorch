import os
import logging

from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer

from data.datasets import CUB200, build_transforms
from engine.trainers import BaseTrainer
from engine.testers import BaseTester


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

        models.cuda()
        train_dataset = CUB200(dataset_root=self.dataset_root, transform=build_transforms(cfg, is_train=True), train=True)
        train_sampler = sampler(train_dataset, self.train_batch_size, num_instances=self.num_instances)
        self.trainer = BaseTrainer(models=models, optimizers=optimizers, lr_schedulers=lr_schedulers, loss_funcs=loss_funcs,
                                   train_dataset=train_dataset, batch_size=self.train_batch_size, dataloader_num_workers=self.dataloader_num_workers,
                                   mining_funcs=mining_funcs, sampler=train_sampler)

        # train_test_dataset = CUB200(dataset_root=self.dataset_root, transform=build_transforms(cfg, is_train=False), train=True)
        # self.tester_train = BaseTester(models=models, test_dataset=train_test_dataset, batch_size=batch_size,
        #                                dataloader_num_workers=self.dataloader_num_workers)

        test_dataset = CUB200(dataset_root=self.dataset_root, transform=build_transforms(cfg, is_train=False), train=False)
        self.tester_test = BaseTester(models=models, test_dataset=test_dataset, batch_size=self.test_batch_size,
                                      dataloader_num_workers=self.dataloader_num_workers)
        self.logger = logging.getLogger('metric.learning')
        self.models = models

        checkpoint_dir = os.path.join(cfg.SAVE_DIR, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpointer = Checkpointer(models, checkpoint_dir, optimizer=optimizers)
        self.periodic_checkpointer = PeriodicCheckpointer(checkpointer=self.checkpointer, period=1, max_iter=10e6)

    def run(self):
        for epoch in range(self.epochs):
            self.logger.info(f"Start Epoch: {epoch}")
            self.trainer.train()
            self.logger.info(f"Test Test-Dataset Epoch: {epoch}")
            accuracy = self.tester_test.test()


            self.periodic_checkpointer.step(epoch)





