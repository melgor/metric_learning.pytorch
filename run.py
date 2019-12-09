import os
from datetime import datetime


from modeling.losses import build_losses
from modeling.miners import build_mining
from data.samplers import build_sampler
from modeling.models import build_model
from modeling.solver.optimizer import build_optimizer
from engine.engine import Engine
from utils.data_logger import setup_logger
from utils import cfg


if __name__ == "__main__":
    dateTimeObj = datetime.now()
    cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, dateTimeObj.strftime("%d-%b-%Y_%H:%M"))
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)

    setup_logger(cfg)
    models = build_model(cfg)
    optimizers = build_optimizer(models)
    loss_funcs = build_losses(cfg)
    mining_funcs = build_mining(cfg)
    sampler = build_sampler(cfg)

    engine = Engine(cfg, models=models, optimizers=optimizers, lr_schedulers=None, loss_funcs=loss_funcs,
                    mining_funcs=mining_funcs, sampler=sampler)

    engine.run()
