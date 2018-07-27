import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from pathlib import Path
import prepare_data

class CaptchaDataset(Dataset):
    def __init__(self, file_names: list, to_augment=False, transform=None, mode='train', img_channels=3):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode
        self.img_channels = img_channels

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        
        if self.mode == 'train':
            img = load_image(img_file_name, img_channels=self.img_channels)
            mask = load_mask(img_file_name)
            img, mask = self.transform(img, mask)
            return to_float_tensor(img, img_channels=self.img_channels), torch.from_numpy(mask).long()
        else:
            img = load_image(img_file_name, img_channels=self.img_channels)
            img = img[2:img.shape[0]-2, 4:img.shape[1]-4] #to (192,96)
            return to_float_tensor(img, img_channels=self.img_channels), str(img_file_name)

def expand(img):
    return np.expand_dims(img, axis=0)

def to_float_tensor(img, img_channels=1):
    if img_channels == 1:
        return torch.from_numpy(img).float()
    else:    
        return torch.from_numpy(np.moveaxis(img, -1, 0)).float()


def load_image(path, img_channels=1):
    img = cv2.imread(str(path), 0 if img_channels == 1 else 1)
    if img_channels == 1:
        return img
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
def load_mask(path):
    factor = prepare_data.digit_factor
    mask_folder = 'masks'

    mask = cv2.imread(str(path).replace('images', mask_folder).replace('jpg', 'png'), 0)

    return (mask / factor).astype(np.uint8)