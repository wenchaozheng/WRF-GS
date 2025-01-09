#! /usr/bin/env python
# -*- coding: utf-8 -*-


import os
import random

import imageio
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from tqdm import tqdm
from einops import rearrange


def split_dataset(datadir, ratio=0.1, dataset_type='rfid'):
    """random shuffle train/test set
    """
    if dataset_type == "rfid":
        spectrum_dir = os.path.join(datadir, 'spectrum')
        spt_names = sorted([f for f in os.listdir(spectrum_dir) if f.endswith('.png')])
        index = [x.split('.')[0] for x in spt_names]
        random.shuffle(index)

    train_len = int(len(index) * ratio)
    train_index = np.array(index[:train_len])
    test_index = np.array(index[train_len:])

    np.savetxt(os.path.join(datadir, "train_index.txt"), train_index, fmt='%s')
    np.savetxt(os.path.join(datadir, "test_index.txt"), test_index, fmt='%s')


class Spectrum_dataset(Dataset):
    """Spectrum dataset class."""
    
    def __init__(self, datadir, indexdir) -> None:
        super().__init__()
        self.datadir = datadir  
        self.tx_pos_dir = os.path.join(datadir, 'tx_pos.csv')  
        self.spectrum_dir = os.path.join(datadir, 'spectrum')  
        self.spt_names = sorted([f for f in os.listdir(self.spectrum_dir) if f.endswith('.png')])       
        self.dataset_index = np.loadtxt(indexdir, dtype=str)  
        self.tx_pos = pd.read_csv(self.tx_pos_dir).values  
        self.n_samples = len(self.dataset_index)  

    def __len__(self):
        return self.n_samples 

    def __getitem__(self, index):
        
        img_name = os.path.join(self.spectrum_dir, self.dataset_index[index] + '.png')
        spectrum = imageio.imread(img_name) / 255.0  
        spectrum = torch.tensor(spectrum, dtype=torch.float32)  

        tx_pos_i = torch.tensor(self.tx_pos[int(self.dataset_index[index]) - 1], dtype=torch.float32)

        return spectrum, tx_pos_i  


dataset_dict = {"rfid": Spectrum_dataset}
