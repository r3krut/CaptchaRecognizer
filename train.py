import argparse
import json
from pathlib import Path
from validation import validation_multi

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.backends.cudnn

from model import UNet11, UNet16
from loss import LossMulti
from dataset import CaptchaDataset
import utils

from prepare_train_val import get_split

from transforms import (DualCompose,
                        ImageOnly,
                        Normalize,
                        ShiftScaleRotate)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--jaccard-weight', default=1, type=float)
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    arg('--fold', type=int, help='fold', default=0)
    arg('--root', default='runs/debug', help='checkpoint root')
    arg('--batch-size', type=int, default=1)
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=8)
    arg('--model', type=str, default='UNet11', choices=['UNet11', 'UNet16'])

    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    #10 digits and background
    num_classes = 11

    if args.model == 'UNet11':
        model = UNet11(num_classes=num_classes, pretrained='vgg')
    elif args.model == 'UNet16':
        model = UNet16(num_classes=num_classes, pretrained='vgg')

    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()

    loss = LossMulti(num_classes=num_classes, jaccard_weight=args.jaccard_weight)

    cudnn.benchmark = True

    def make_loader(file_names, shuffle=False, transform=None):
        return DataLoader(
            dataset=CaptchaDataset(file_names, transform=transform),
            shuffle=shuffle,
            num_workers=args.workers,
            batch_size=args.batch_size,
            pin_memory=torch.cuda.is_available()
        )

    train_file_names, val_file_names = get_split(args.fold)

    print('num_train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))

    train_transform = DualCompose([ShiftScaleRotate(), ImageOnly(Normalize())])

    val_transform = DualCompose([ImageOnly(Normalize())])

    train_loader = make_loader(train_file_names, shuffle=True, transform=train_transform)
    valid_loader = make_loader(val_file_names, transform=val_transform)

    root.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))

    valid = validation_multi

    utils.train(
        init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
        args=args,
        model=model,
        criterion=loss,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=valid,
        fold=args.fold,
        num_classes=num_classes
    )

if __name__ == '__main__':
    main()