import torch


def build_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr_mul = 1.0
        if "backbone" in key:
            lr_mul = 0.1
        params += [{"params": [value], "lr_mul": lr_mul}]
    optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params,
                                                                lr=cfg.SOLVER.BASE_LR,
                                                                weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    return optimizer
