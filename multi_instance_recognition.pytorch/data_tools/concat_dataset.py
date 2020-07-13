 # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import numpy as np
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset
from .instance_set import InstanceSet, my_collate
from torch.utils.data import Dataset, DataLoader


class ConcatDataset(_ConcatDataset):
    """
    Same as torch.utils.data.dataset.ConcatDataset, but exposes an extra
    method for querying the sizes of the image
    """

    def get_idxs(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def get_img_info(self, idx):
        dataset_idx, sample_idx = self.get_idxs(idx)
        return self.datasets[dataset_idx].get_img_info(sample_idx)

class MixDataset(object):
    def __init__(self,datasets,ratios):
        self.datasets=datasets
        self.ratios=ratios
        self.lengths=[]
        for dataset in self.datasets:
            self.lengths.append(len(dataset))
        self.lengths=np.array(self.lengths)
        self.seperate_inds=[]
        s=0
        for i in self.ratios[:-1]:
            s+=i
            self.seperate_inds.append(s)

    def __len__(self):
       return self.lengths.sum()
       
    def __getitem__(self, item):
        i=np.random.rand()
        ind=bisect.bisect_right(self.seperate_inds,i)
        b_ind=np.random.randint(self.lengths[ind])
        return self.datasets[ind][b_ind]
    #def get_img_info(self,idx):


def build_dataset(data_cfg):
    if isinstance(data_cfg['data_type'], list):
        datasets_list = [
            InstanceSet(data_cfg, data_type) for data_type in data_cfg['data_type']
        ]
        mix_dataset = MixDataset(datasets_list, data_cfg['ratios'])
        return mix_dataset
    else:
        return InstanceSet(data_cfg, data_cfg['data_type'])


def build_dataloader(data_cfg):
    dataset = build_dataset(data_cfg)
    data_loader = DataLoader(dataset, batch_size=data_cfg['batch_size'], collate_fn=my_collate, shuffle=True,
                             num_workers=data_cfg['num_works'])
    return data_loader


