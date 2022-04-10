import os
from functools import partial
import numpy as np

import rasterio
import random
import pandas as pd
import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

AUTOTUNE = tf.data.AUTOTUNE


# Calculate mean for each channel over all pixels for training set; for validation and test set you need to take
# mean and std of training set as well as in real case scenarios you don't know them beforehand to calculate them
def calc_mean(img_dir: str, img_list: list, input_height: int, input_width: int, clipping_values: list, channels: list):
    '''
    Calculate mean pixel values per channel over all input images
    Args:
        img_dir (str): Path to image data
        input_height (int): pixel height of input
        input_width (int): pixel width of input
        clipping_values (list): interval of min and max values for clipping
        channels (list):  Channels to use; [] if all channels are to be used

    Returns (np.array): Means of pixel values per channel

    '''

    # Ensures that only tif data are in the current directory, if not it will throw an error
    assert all([i.endswith('.tif') for i in img_list])

    # Variable to save the summation of the pixels values
    sum_arr = 0
    # Count of pixels
    sum_pixel = input_height * input_width * len(img_list)

    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name)
        with rasterio.open(img_path) as img:
            # Read in image with all channels
            if len(channels) == 0:
                array = img.read()
            # Only read defined channels of the image
            else:
                array = img.read(channels)

        array = array.astype('float32')
        # Replace NaN with zeros (for calculation required)
        array[np.isnan(array)] = 0
        # Clipping
        array = np.clip(array, a_min=clipping_values[0], a_max=clipping_values[1])
        # Ensure that that all arrays have the same size
        array = np.resize(array, (array.shape[0], input_width, input_height))
        # Add up the sum over the input and height for each channel separately
        sum_arr += array.sum(axis=(1, 2))

    # Calculate mean for each channel
    means = sum_arr / sum_pixel

    # return mean per channel(#mean == # channels)
    return means


# Calculate standard deviation (note:mean function has to be executed beforehand as it is required as input)
def calc_std(means, img_dir: str, img_list: list, input_height: str, input_width: str, clipping_values: list,
             channels: list):
    '''
    Calculate standard deviation values per channel over all input images
    Args:
        means (np.array): Result of calc_mean: mean pixel values for each channel
        img_dir (str): Path to image data
        input_height (int): pixel height of input
        input_width (int): pixel width of input
        clipping_values (list): interval of min and max values for clipping
        channels (list):  Channels to use; [] if all channels are to be used

    Returns (np.array): Standard deviation of pixel values per channel

    '''

    # Ensure all data are tif data in current directory
    assert all([i.endswith('.tif') for i in img_list])

    sum_arr = 0
    # Count of pixels
    sum_pixel = input_height * input_width * len(img_list)

    # Sum each array/image up (channelwise) with each other, after substracting mean(s) from each array and take it ^2
    # each one seperately -> Check standard deviatin equation
    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name)
        with rasterio.open(img_path) as img:
            if len(channels) == 0:
                # Read in all channels
                array = img.read()
            else:
                # Only read in predefined channels
                array = img.read(channels)

        array = array.astype('float32')
        array[np.isnan(array)] = 0

        # Clipping
        array = np.clip(array, a_min=clipping_values[0], a_max=clipping_values[1])
        # Ensure that that all arrays have the same size
        array = np.resize(array, (array.shape[0], input_width, input_height))

        array = np.power(array.transpose(1, 2, 0) - means, 2).transpose(2, 0, 1)
        sum_arr += array.sum(axis=(1, 2))

    # Second part of equation
    stds = np.sqrt(sum_arr / sum_pixel)

    return stds


def create_splits(img_dir: str, wealth_path: str, urban_rural: str, subset=False):
    '''
    Create train/val and testsplit for Cross Validation.
    Args:
        img_dir (str): path to image directory
        wealth_path (str): path to label csv files
        urban_rural (str): on of ['u','r','ur'] to choose whether to use only urban/rural clusters or all data
        subset (bool): Whether or not to use a subset (for testing)

    Returns:
        X_train_val (list): list containing filenames for train and validation split
        X_test (list): list containing filenames for test split
        y_train_val (np.ndarray): numpy array containing asset wealth (label values) for train and validation split
        y_test (np.ndarray): numpy array containing asset wealth (label values) for test split
    '''
    if subset:
        if urban_rural == 'ur' or urban_rural == 'UR':
            img_list = [img for img in os.listdir(img_dir) if img.endswith('.tif')][:50]
        elif urban_rural == 'u' or urban_rural == 'U':
            img_list = [img for img in os.listdir(img_dir) if img.endswith('u_2.0.tif')][:50]
        elif urban_rural == 'r' or urban_rural == 'R':
            img_list = [img for img in os.listdir(img_dir) if img.endswith('r_10.0.tif')][:50]
        else:
            raise ValueError(f'{urban_rural} is not a valid argument.')
    else:
        if urban_rural == 'ur' or urban_rural == 'UR':
            img_list = [img for img in os.listdir(img_dir) if img.endswith('.tif')]
        elif urban_rural == 'u' or urban_rural == 'U':
            img_list = [img for img in os.listdir(img_dir) if img.endswith('u_2.0.tif')]
        elif urban_rural == 'r' or urban_rural == 'R':
            img_list = [img for img in os.listdir(img_dir) if img.endswith('r_10.0.tif')]
        else:
            raise ValueError(f'{urban_rural} is not a valid argument.')
    wealth_df = combine_wealth_dfs(wealth_path)
    y = []
    for img in img_list:
        y.append(get_label_for_img(wealth_df, img).WEALTH_INDEX.iloc[0])
    y = np.array(y)
    X_train_val, X_test, y_train_val, y_test = train_test_split(img_list, y, test_size=0.20, random_state=42)

    return X_train_val, X_test, y_train_val, y_test


def generator(img_dir: str, X: list, y: np.ndarray, batch_size: int, input_height: int, input_width: int,
              clipping_values: list, channel_size: int, channels: list):
    '''
    Data generator to generate label and feature batches.
    Args:
        img_dir (str): Path to img directory
        X (list): List containing filenames of split
        y (np.ndarray): Array containing the label values of train/val split
        batch_size (int): Size of training batches
        input_height (int): pixel height of input
        input_width (int): pixel width of input

    Returns
        batch_x (np.ndarray): feature batch
        batch_y (np.ndarray): label batch
    '''
    assert all([i.endswith('.tif') for i in X])
    means = calc_mean(img_dir=img_dir, img_list=X, input_height=input_height,
                      input_width=input_width, clipping_values=clipping_values, channels=channels)
    stds = calc_std(means=means, img_dir=img_dir, img_list=X, input_height=input_height,
                    input_width=input_width, clipping_values=clipping_values, channels=channels)

    # generate batches (x : input, y: label)
    batch_x = np.zeros(shape=(batch_size, channel_size, input_height, input_width))
    batch_y = np.zeros(shape=(batch_size,), dtype=float)

    # Iterator
    batch_ele = 0
    for index, img in tqdm(enumerate(X)):
        array = normalize_resize(os.path.join(img_dir, img), input_height, input_width, clipping_values, means, stds)
        # Add to batch
        batch_x[batch_ele] = array

        # Get corresponding label y
        batch_y[batch_ele] = y[index]

        # Check if batch is already full (Note: Index in batch array is from 0...4 hence we need to add +1 to batch_ele)
        if (batch_ele + 1) == batch_size:
            batch_x = batch_x.transpose(0, 2, 3, 1)
            # Return of batch_x,batch_y
            yield batch_x.astype(np.float64), batch_y.astype(np.float32)
            # Reset settings -> Start of next batch generation
            batch_ele = 0
            batch_x = np.zeros(shape=(batch_size, channel_size, input_height, input_width))
            batch_y = np.zeros(shape=(batch_size,), dtype=float)
        else:
            batch_ele += 1
    return batch_x, batch_y


def combine_wealth_dfs(wealth_csv_path: str):
    '''
    Combines all label csv files to one.
    Args:
        wealth_csv_path (str): path to label csv data

    Returns:
        complete_wealth_df (pd.DataFrame): Pandas Dataframe containing all clusters
    '''

    wealth_files = glob.glob(wealth_csv_path + "/*.csv")
    wealth_df_list = []

    for filename in wealth_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        wealth_df_list.append(df)

    complete_wealth_df = pd.concat(wealth_df_list, axis=0, ignore_index=True)
    return complete_wealth_df


def get_label_for_img(wealth_df: pd.DataFrame, img_filename: str):
    '''
    Get label data for a cluster based on the filename.
    Args:
        wealth_csv: Path to DHS Wealth CSV File
        img_dir: Path to Image Directory
        urban_rural

    Returns:
        wealth_sentinel_df: Dataframe including dhs survey data, geo coordinates and img file destination

    '''

    img_info_df = pd.DataFrame([list(get_img_coordinates(img_filename)) + [img_filename]],
                               columns=['LATNUM', 'LONGNUM', 'Filename'])

    wealth_df['LATNUM'] = wealth_df.LATNUM.apply(lambda x: truncate(x, 4))
    wealth_df['LONGNUM'] = wealth_df.LONGNUM.apply(lambda x: truncate(x, 4))
    label = wealth_df.merge(img_info_df, on=['LATNUM', 'LONGNUM'])[['DHSYEAR', 'DHSCLUST', 'URBAN_RURA', 'LATNUM',
                                                                    'LONGNUM', 'WEALTH_INDEX', 'SURVEY_YEAR', 'COUNTRY',
                                                                    'Filename']]

    return label


def get_img_coordinates(img: str):
    '''
    Extract the cluster coordinates from a given filename.
    Args:
        img (str): Filename of Image

    Returns:
        str, str : Latitude, Longitude

    '''
    return img.split('_')[0], img.split('_')[1]


def truncate(f, n):
    '''
    Truncates a float f to n decimal places without rounding
    Args:
        f: float value
        n: number of decimal places

    Returns:

    '''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d + '0' * n)[:n]])