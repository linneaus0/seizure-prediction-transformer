from PIL import Image
import torch
from torch.utils.data import Dataset
import json
import os
import os.path
import numpy as np
from utils.load_signals import PrepData
from utils.prep_data import train_val_loo_split, train_val_test_split

class CHBMIT_DataSet(Dataset):
    """CHB-MIT单个病人数据集"""
    def __init__(self, target, settings, num_classes=2):
        self.target = target
        self.num_classes = num_classes
        self.settings = settings

    def __getitem__(self):
        # x:data [Channel, Time, Freq] y:label
        ictal_X, ictal_y = PrepData(self.target, type='ictal', settings=self.settings).apply()
        self.len = len(ictal_y)
        interictal_X, interictal_y = PrepData(self.target, type='interictal', settings=self.settings).apply()
        loo_folds = train_val_loo_split(ictal_X, ictal_y, interictal_X, interictal_y, 0.25)

        return loo_folds

    def __len__(self):
        return self.len

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
