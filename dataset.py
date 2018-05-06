"""
This code was taken from https://github.com/ternaus/robot-surgery-segmentation/dataset.py 
with slightly modifications 
"""

import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from pathlib import Path
import prepare_data

data_path = Path('/home/r3krut/DataSets/NalogCaptchaDataTraining/CaptchaRecognition/train')

class CaptchaDataset(Dataset):
    def __init__(self, file_names: list, to_augment=False, transform=None, mode='train'):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        img = load_image(img_file_name)
        mask = load_mask(img_file_name)

        img, mask = self.transform(img, mask)

        if self.mode == 'train':
            return to_float_tensor(img), torch.from_numpy(mask).long()
        else:
            return to_float_tensor(img), str(img_file_name)


def to_float_tensor(img):
    return torch.from_numpy(np.moveaxis(img, -1, 0)).float()


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path):
    factor = prepare_data.digit_factor
    mask_folder = 'masks'

    mask = cv2.imread(str(path).replace('images', mask_folder).replace('jpg', 'png'), 0)

    return (mask / factor).astype(np.uint8)