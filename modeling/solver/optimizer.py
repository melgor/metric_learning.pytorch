import torch


def build_optimizer(model, name="Adam", lr=0.00001, wd=0.0005):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr_mul = 1.0
        if "backbone" in key:
            lr_mul = 0.1
        params += [{"params": [value], "lr_mul": lr_mul}]
    optimizer = getattr(torch.optim, name)(params,
                                           lr=lr,
                                           weight_decay=wd)
    return optimizer
