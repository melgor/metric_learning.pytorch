import os
from datetime import datetime
from contextlib import redirect_stdout

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

    # save current config to output directory
    with open(f"{cfg.SAVE_DIR}/config.yml", 'w') as f:
        with redirect_stdout(f):
            print(cfg.dump())

    setup_logger(cfg)
    models = build_model(cfg)
    optimizers = build_optimizer(cfg, models)
    loss_funcs = build_losses(cfg)
    mining_funcs = build_mining(cfg)
    sampler = build_sampler(cfg)

    engine = Engine(cfg, models=models, optimizers=optimizers, lr_schedulers=None, loss_funcs=loss_funcs,
                    mining_funcs=mining_funcs, sampler=sampler)

    engine.run()
