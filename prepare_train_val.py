from prepare_data import data_train
import numpy as np

def get_split(fold):
    folds = {0: np.arange(0,20),
             1: np.arange(20,40),
             2: np.arange(40,60),
             3: np.arange(60,80),
             4: np.arange(80,100),
             5: np.arange(100,122)}

    train_path = data_path / 'images'

    train_file_names = []
    val_file_names = []

    for captcha_id in range(0, 121):
        if captcha_id in folds[fold]:
            val_file_names.append(str((train_path / ('img_' + str(captcha_id) + '.jpg'))))
        else:
            train_file_names.append(str((train_path / ('img_' + str(captcha_id) + '.jpg'))))

    return train_file_names, val_file_names