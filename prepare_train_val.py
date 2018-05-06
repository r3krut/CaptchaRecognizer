from dataset import data_path


def get_split(fold):
    folds = {0: [0, 19],
             1: [20, 39],
             2: [40, 59],
             3: [60, 79],
             4: [80, 99],
             5: [100, 121]}

    train_path = data_path / 'images'

    train_file_names = []
    val_file_names = []

    for captcha_id in range(0, 121):
        if captcha_id in folds[fold]:
            val_file_names += (train_path / ('img_' + str(captcha_id) + '.jpg'))
        else:
            train_file_names += (train_path / ('img_' + str(captcha_id) + '.jpg'))

    return train_file_names, val_file_names