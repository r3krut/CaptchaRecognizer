"""
This code was taken from https://github.com/ternaus/robot-surgery-segmentation/generate_masks.py 
with slightly modifications 
"""

"""
Script generates predictions, splitting original images into tiles, and assembling prediction back together
"""
import argparse
from prepare_train_val import get_split
from dataset import CaptchaDataset
import dataset as ds
from prepare_data import data_tests
import cv2 as cv
from model import UNet16, UNet11
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import utils
import prepare_data
from torch.utils.data import DataLoader
from torch.nn import functional as F

from transforms import (ImageOnly,
                        Normalize,
                        ShiftScaleRotate,
                        DualCompose)

img_transform = DualCompose([
    ImageOnly(Normalize())
])

path_to_test = data_tests

def get_model(model_path, model_type='unet11'):
    """

    :param model_path:
    :param model_type: 'UNet16', 'UNet11'
    :return:
    """
    num_classes = 11

    if model_type == 'unet16':
        model = UNet16(num_classes=num_classes)
    elif model_type == 'unet11':
        model = UNet11(num_classes=num_classes)
    
    state = torch.load(str(model_path))
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(state)

    if torch.cuda.is_available():
        return model.cuda()

    model.eval()

    return model

def colored_masks(img):
    height = img.shape[0]
    width = img.shape[1]
    colored_mask = np.zeros((height, width, 3))

    for h in range(0, width-1):
        for w in range(0, height-1):
            pixel = img[w, h]
            if pixel == 20:                 #Color(BGR) for digit 0
                colored_mask[w,h] = [255,0,0]
            if pixel == 40:                 #Color(BGR) for digit 1
                colored_mask[w,h] = [0,255,0]
            if pixel == 60:                 #2
                colored_mask[w,h] = [255,255,0]
            if pixel == 80:                 #3
                colored_mask[w,h] = [0,0,255]
            if pixel == 100:                #4
                colored_mask[w,h] = [255,0,255]
            if pixel == 120:                #5
                colored_mask[w,h] = [0,255,255]
            if pixel == 140:                #6
                colored_mask[w,h] = [0,150,255]
            if pixel == 160:                #7
                colored_mask[w,h] = [115,0,255]
            if pixel == 180:                #8
                colored_mask[w,h] = [0,100,0]
            if pixel == 200:                #9
                colored_mask[w,h] = [255,140,140]
    colored_mask = colored_mask.astype(np.uint8)
    return colored_mask


def predict(model, from_file_names, batch_size: int, to_path, single_predict=True):
    loader = DataLoader(
        dataset=CaptchaDataset(from_file_names, transform=img_transform, mode='predict'),
        shuffle=False,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=torch.cuda.is_available()
    )

    for batch_num, (inputs, paths) in enumerate(loader):
        inputs = utils.variable(inputs, volatile=True)

        outputs = model(inputs)

        for i, image_name in enumerate(paths):
            factor = prepare_data.digit_factor
            t_mask = (outputs[i].data.cpu().numpy().argmax(axis=0) * factor).astype(np.uint8)

            h, w = t_mask.shape

            full_mask = np.zeros((100, 200)) #to (100, 200)
            full_mask[2:2 + h, 4:4 + w] = t_mask

            full_mask = colored_masks(full_mask)

            (to_path).mkdir(exist_ok=True, parents=True)

            if single_predict:
                cv.imwrite(str(to_path / 'mask.png'), full_mask)
            else:
                cv.imwrite(str(to_path / (str(Path(image_name).stem) + '.png')), full_mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_path', type=str, default='data/models/unet11', help='path to model folder')
    arg('--model_type', type=str, default='unet11', help='network architecture',
        choices=['unet11', 'unet16'])
    arg('--output_path', type=str, help='path to save images', default='.')
    arg('--batch-size', type=int, default=4)
    arg('--fold', type=int, default=None, choices=[None, 0, 1, 2, 3, 4, 5, -1], help='-1: all folds')
    arg('--workers', type=int, default=8)

    args = parser.parse_args()

    if args.fold is None:
        file_names = list(path_to_test.glob('*'))
        
        model = get_model(str(Path(args.model_path).joinpath('model_{model}.pt'.format(model=args.model_type))),
                              model_type=args.model_type)

        print('Num file_names = {}'.format(len(file_names)))

        output_path = Path(args.output_path)
        output_path.mkdir(exist_ok=True, parents=True)

        predict(model, file_names, args.batch_size, output_path, single_predict=False)

    elif args.fold == -1:
        for fold in [0, 1, 2, 3, 4, 5]:
            _, file_names = get_split(fold)
            model = get_model(str(Path(args.model_path).joinpath('model_{model}.pt'.format(model=args.model_type))),
                              model_type=args.model_type)

            print('Num file_names = {}'.format(len(file_names)))

            output_path = Path(args.output_path)
            output_path.mkdir(exist_ok=True, parents=True)

            predict(model, file_names, args.batch_size, output_path, single_predict=False)
    else:
        _, file_names = get_split(args.fold)
        model = get_model(str(Path(args.model_path).joinpath('model_{model}.pt'.format(model=args.model_type))),
                          model_type=args.model_type)

        print('Num file_names = {}'.format(len(file_names)))

        output_path = Path(args.output_path)
        output_path.mkdir(exist_ok=True, parents=True)

        predict(model, file_names, args.batch_size, output_path, single_predict=False)