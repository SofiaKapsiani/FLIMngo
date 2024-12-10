import os
import numpy as np
import h5py

import torch
from torch.utils.data import Dataset

class HDF5Dataset(Dataset):
    """
    imgs_dir: directory where images are stored
    img_internal_path: H5 internal path to the raw FLIM data
    label_internal_path: H5 internal path to the lifetime maps
    
    Raw flim data will be loaded as: (T, Y, X), and the labels as: (Y, X)
    """
    def __init__(self, imgs_dir, img_internal_path='tpsf', label_internal_path='tau_maps',):
        self.imgs_dir = imgs_dir
        self.img_ids = os.listdir(imgs_dir)
        self.img_internal_path = img_internal_path
        self.label_internal_path = label_internal_path

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.imgs_dir, self.img_ids[idx])
        f= h5py.File(img_path, 'r')
        image = np.asarray(f.get(self.img_internal_path))
        label = np.asarray(f.get(self.label_internal_path))

        image = torch.from_numpy(image).permute(0, 2, 1)
        image = image.float()

        label = torch.from_numpy(label).permute(1, 0).float()
       
        return image, label

