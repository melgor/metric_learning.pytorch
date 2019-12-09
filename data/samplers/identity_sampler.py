from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import Sampler

from .registry import SAMPLER


@SAMPLER.register()
class IdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample M instances, therefore batch size is N*M.
    Args:
    - data_source : data source with class variable target
    - batch_size: number of examples in a batch.
    - num_instances: number of instances per identity in a batch.


    Note: Sample assume that each class has more examples than num_instances
          In other case, training will fail
    """

    def __init__(self, data_source, batch_size: int, num_instances: int = 1):
        assert hasattr(data_source, 'targets')
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances

        # create dict where key is class idx and value in idx of elements from that class
        self.index_dic = defaultdict(list)
        for index, pid in enumerate(data_source.targets):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_iter = len(self.data_source) // self.batch_size
        assert all(len(val) > self.num_instances for val in self.index_dic.values()), \
            "Some classes have lower number of instances than num_instances, please make num_instances value smaller"

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        for iter_num in range(self.num_iter):
            # in each iter take random classes.
            # In better implementation we should care that each class is taken same numer of time
            indices = torch.randperm(self.num_pids_per_batch)

            for i in indices:
                pid = self.pids[i]
                t = self.index_dic[pid]
                # choice random examples from class. In case of lower number of example per class, copy instances
                t = np.random.choice(t, size=self.num_instances, replace=False if len(t) > self.num_instances else True)
                yield from t
