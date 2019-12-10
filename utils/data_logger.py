import logging
import sys
from os import path
import torch
from logging.handlers import WatchedFileHandler
from collections import defaultdict

from fvcore.common.history_buffer import HistoryBuffer

def find_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def setup_logger(cfg):
    logger = logging.getLogger(cfg.LOGGER.NAME)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s')
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = WatchedFileHandler(path.join(cfg.SAVE_DIR, "task.log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)


class MetricLogger(object):
    def __init__(self, name="", delimiter="\t"):
        self.meters = defaultdict(HistoryBuffer)
        self.name = name
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getitem__(self, item):
        return self.meters[item]

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg)
            )
        return self.delimiter.join(loss_str)

    def avg(self, window_size=1):
        return {f"{self.name}_{key}": value.avg(window_size=window_size) for key, value in self.meters.items()}


