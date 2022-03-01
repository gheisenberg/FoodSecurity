import os
import shutil
from data_utils import combine_wealth_dfs, truncate

import pandas as pd
import glob
from sklearn.model_selection import train_test_split
import random


def get_img_coordinates(img: str):
    '''

    Args:
        img: Filename of Image

    Returns:
        Latitude, Longitude

    '''
    return img.split('_')[0], img.split('_')[1]


def get_labels_df_for_img(img_dir: str):
    '''

    Args:
        wealth_csv: Path to DHS Wealth CSV File
        img_dir: Path to Image Directory
        urban_rural

    Returns:
        wealth_sentinel_df: Dataframe including dhs survey data, geo coordinates and img file destination

    '''
    img_list = os.listdir(img_dir)

    img_info_df = pd.DataFrame([list(get_img_coordinates(img)) + [img] for img in img_list],
                               columns=['LATNUM', 'LONGNUM', 'Filename'])
    img_info_df['file_path'] = img_dir

    wealth_df = combine_wealth_dfs('/home/stoermer/Sentinel/gps_csv/')
    wealth_df['LATNUM'] = wealth_df.LATNUM.apply(lambda x: truncate(x, 4))
    wealth_df['LONGNUM'] = wealth_df.LONGNUM.apply(lambda x: truncate(x, 4))
    wealth_sentinel_df = wealth_df.merge(img_info_df, on=['LATNUM', 'LONGNUM'])

    return wealth_sentinel_df


def get_label_df(wealth_csv_path: str, img_dir: str, x_train: list, x_val: list, x_test: list):
    '''

    Args:
        wealth_csv: Path to DHS Wealth CSV File
        img_dir: Path to Image Directory
        urban_rural

    Returns:
        label_df: Dataframe including dhs survey data, geo coordinates and img file destination

    '''
    x_train_df = pd.DataFrame(x_train, columns=['filename'])
    x_train_df['split'] = 'train'
    x_train_df['file_path'] = os.path.join(img_dir, 'subset_train')

    x_val_df = pd.DataFrame(x_val, columns=['filename'])
    x_val_df['split'] = 'val'
    x_val_df['file_path'] = os.path.join(img_dir, 'subset_val')

    x_test_df = pd.DataFrame(x_test, columns=['filename'])
    x_test_df['split'] = 'test'
    x_test_df['file_path'] = os.path.join(img_dir, 'subset_test')

    img_df = pd.concat([x_train_df, x_test_df, x_val_df]).reset_index(drop=True)
    img_df["LATNUM"], img_df["LONGNUM"] = zip(*img_df["filename"].map(get_img_coordinates))

    wealth_df = combine_wealth_dfs(wealth_csv_path)
    wealth_df['LATNUM'] = wealth_df.LATNUM.apply(lambda x: truncate(x, 4))
    wealth_df['LONGNUM'] = wealth_df.LONGNUM.apply(lambda x: truncate(x, 4))
    label_df = wealth_df.merge(img_df, on=['LATNUM', 'LONGNUM'])[['WEALTH_INDEX', 'filename']]

    return label_df


def ratio_val_to_test(val: float, test: float):
    total = val + test
    one_perc = 100.00 / total
    val_ratio = one_perc * val * 0.01

    return val_ratio


def created_data_sets(img_dir: str, split_size: list, subset: bool):
    '''

    Args:
        img_dir: path to urban or rural  sentinel2 image directory
        split_size: precentage of train, val, test from total e.g. [60,20,20]

    Returns:

    '''
    print('Creating splits with split sizes:', split_size)
    img_list = [img for img in os.listdir(img_dir) if img.endswith('.tif')]
    if subset:
        img_list = random.sample(img_list, 100)

        # Create  validation, training und test folder

        train_dir = os.path.join(img_dir, 'subset_train')
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        val_dir = os.path.join(img_dir, 'subset_val')
        if not os.path.exists(val_dir):
            os.mkdir(val_dir)

        test_dir = os.path.join(img_dir, 'subset_test')
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
    else:
        train_dir = os.path.join(img_dir, 'train')
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        val_dir = os.path.join(img_dir, 'val')
        if not os.path.exists(val_dir):
            os.mkdir(val_dir)

        test_dir = os.path.join(img_dir, 'test')
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)

    # Split into the data sets and move them to their respective folder

    X_train, X_rem = train_test_split(img_list, train_size=split_size[0] / 100)

    X_val, X_test = train_test_split(X_rem, train_size=ratio_val_to_test(split_size[1] / 100, split_size[2] / 100))

    for img in X_train:
        img_path = os.path.join(img_dir, img)
        train_img = os.path.join(train_dir, img)
        shutil.copyfile(img_path, train_img)
    print(f'Train set finished. Contains {len(X_train)} images.')

    for img in X_val:
        img_path = os.path.join(img_dir, img)
        val_img = os.path.join(val_dir, img)
        shutil.copyfile(img_path, val_img)
    print(f'Validation set finished. Contains {len(X_val)} images.')

    for img in X_test:
        img_path = os.path.join(img_dir, img)
        test_img = os.path.join(test_dir, img)
        shutil.copyfile(img_path, test_img)
    print(f'Test set finished. Contains {len(X_test)} images.')

    return X_train, X_val, X_test


def main(img_dir: str, splits: list, subset: bool, wealth_csv_path: str):
    X_train, X_val, X_test = created_data_sets(img_dir=img_dir,
                                               split_size=splits,
                                               subset=subset)
    label_df = get_label_df(wealth_csv_path=wealth_csv_path,
                            img_dir=img_dir,
                            x_train=X_train,
                            x_val=X_val,
                            x_test=X_test)
    if subset:
        label_df.to_csv(os.path.join(img_dir, 'subset_labels.csv'))
    else:
        label_df.to_csv(os.path.join(img_dir, 'labels.csv'))


if __name__ == '__main__':
    main(img_dir='/mnt/datadisk/data/Sentinel2/preprocessed/asset/rural',
         splits=[60, 20, 20],
         subset=True,
         wealth_csv_path='/home/stoermer/Sentinel/gps_csv'
         )

