import sys

sys.path.append("..")

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from osgeo import osr
from osgeo import gdal
from src.data_utils import calc_mean
from src.data_utils import calc_std

from src.config import clipping_values

def slice_to_input_size(array: np.ndarray, input_height: int, input_width: int):
    '''

    Args:
        array (np.ndarray): Numpy Array containing Image Data
        input_height (int): Uniform Image Height to slice to
        input_width (int):  Uniform Image Width to slice to

    Returns:
        array (np.array): Numpy Array containing Image Data in shape of Input Height/Width and Bandwidth
    '''
    if len(array.shape) == 3:
        outer_pixels_y = int((array.shape[1] - input_height) / 2)
        outer_pixels_x = int((array.shape[2] - input_width) / 2)
        array = array[:, outer_pixels_y:-outer_pixels_y, outer_pixels_x:-outer_pixels_x]
        if array.shape[1] == input_height + 1:
            array = array[:, :-1, :]
        if array.shape[2] == input_width + 1:
            array = array[:, :, :-1]
    elif len(array.shape) == 2:
        outer_pixels_y = int((array.shape[0] - input_height) / 2)
        outer_pixels_x = int((array.shape[1] - input_width) / 2)
        array = array[outer_pixels_y:-outer_pixels_y, outer_pixels_x:-outer_pixels_x]
        if array.shape[0] == input_height + 1:
            array = array[:-1, :]
        if array.shape[1] == input_width + 1:
            array = array[:, :-1]
    return array


def standardize_resize(img: str, img_path: str,  input_height: str, input_width: str, clipping_values: list,
                     means=False, stds=False, add_img_path=False, standardize=False):
    '''
    Standardize and Resize GeoTIFFs.
    Standardization is performed per Band using Standard Scaler.
    Resizing is performed by slicing to the Center of the Image in Shape of provided Input Size.
    For VIIRS Images, the Band is tripled to fit RGB Input Shape of common CNNs.
    Standardized and Resized Images are stored in a new GeoTIFF.
    Args:
        img (str):                  Filename of Image to Normalize and Resize
        img_path (str):             Path to Image Data
        input_height (int):         Desired Input Height
        input_width (int):          Desired Input Width
        clipping_values (list):     Interval of Min and Max Values for Clipping
        means (bool/np.ndarray):         Optional: Result of calc_mean: Mean of Pixel Values for each Channel
        stds (bool/np.ndarray):          Optional: Result of calc_mean: Standard Deviation of Pixel Values for each Channel
        add_img_path (bool/str):    Optional: Path to Image Data to add (eg. for combining Sentinel2 and VIIRS)
        standardize (bool):                Optional: Whether or not to standardize Image Data (e.g. standardization is not needed when already normalized Sentinel2 and VIIRS data are merged)

    '''
    ds = gdal.Open(os.path.join(img_path, img))
    geotransform = ds.GetGeoTransform()
    array = ds.ReadAsArray()

    array[np.isnan(array)] = 0
    assert not np.any(np.isnan(array)), "Float"

    if add_img_path:
        ds = gdal.Open(os.path.join(add_img_path, img))
        geotransform = ds.GetGeoTransform()
        array_to_add = ds.ReadAsArray()

        array_to_add[np.isnan(array_to_add)] = 0
        assert not np.any(np.isnan(array_to_add)), "Float"
    # Clipping
    array = np.clip(array, a_min=clipping_values[0], a_max=clipping_values[1])

    assert not np.any(np.isnan(array)), "After clipping"
    # Normalize the array
    if standardize:
        if 'Sentinel' in img_path:
            array = ((array.transpose(1, 2, 0) - means) / stds).transpose(2, 0, 1)
        elif 'VIIRS' in img_path:
            array = ((array.transpose(0, 1) - means) / stds).transpose(1, 0)
        assert not np.any(np.isnan(array)), "Normalize"

    # Resize the array to ensure that all image arrays have the same size
    if len(array.shape) == 3:
        if array.shape != (array.shape[0], input_height, input_width):
            if array.shape[1] > input_height:
                array = slice_to_input_size(array, input_height, input_width)

        assert array.shape == (array.shape[0], input_height, input_width)
    elif len(array.shape) == 2:
        if array.shape != (input_height, input_width):
            if array.shape[0] > input_height:
                array = slice_to_input_size(array, input_height, input_width)

        assert array.shape == (input_height, input_width)
    # return normalized and resized image array

    # Create new geotiff
    if add_img_path:
        dst_ds = gdal.GetDriverByName('GTiff').Create(
            os.path.join('/mnt/datadisk/data/VIIRS_Sentinel2/preprocessed/asset/',
                         img_path.split('/')[-2],
                         img_path.split('/')[-1],
                         img), input_height, input_width,len(array) + 1, gdal.GDT_Byte)
    if 'Sentinel' in img_path:
        dst_ds = gdal.GetDriverByName('GTiff').Create(os.path.join(img_path, img), input_height, input_width,
                                                      len(array), gdal.GDT_Byte)
    elif 'VIIRS' in img_path:
        dst_ds = gdal.GetDriverByName('GTiff').Create(os.path.join(img_path, img), input_height, input_width, 3,
                                                      gdal.GDT_Byte)
    dst_ds.SetGeoTransform(geotransform)  # specify coords
    srs = osr.SpatialReference()  # establish encoding
    srs.ImportFromEPSG(4326)  # WGS84 lat/long
    dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
    if 'Sentinel' in img_path:
        for i in range(len(array)):
            dst_ds.GetRasterBand(i + 1).WriteArray(array[i])  # write band to the raster
    elif 'VIIRS' in img_path:
        array = np.concatenate((np.array([array]),) * 3, axis=0)

        for i in range(len(array)):
            dst_ds.GetRasterBand(i + 1).WriteArray(array[i])  # write band to the raster
    if add_img_path:
        for i in range(len(array)):
            dst_ds.GetRasterBand(i + 1).WriteArray(array[i])  # write band to the raster
        dst_ds.GetRasterBand(i + 2).WriteArray(array_to_add[0])  # write (viirs) band to the raster
    dst_ds.FlushCache()  # write to disk
    dst_ds = None


def main(img_path: str, ur: str, year: str, input_height: str, input_width: str,
         clipping_values: list, channels: list, add_img_path=False, standardize=False):
    '''

    Args:
        img_path (str):             Path to Image Data
        ur (str):                   'u' for urban, 'r' for rural
        year (str):                 timespan (2012_2014 / 2016_2020) or all data
        input_height (int):         Desired Input Height
        input_width (int):          Desired Input Width
        clipping_values (list):     Interval of Min and Max Values for Clipping
        channels (list):            List of Channels to use. [] to use all channels.
        add_img_path (bool/str):    Optional: Path to Image Data to add (eg. for combining Sentinel2 and VIIRS)
        standardize:                Optional: Whether or not to standardize Image Data (e.g. standardization is not needed when already
                                    normalized Sentinel2 and VIIRS data are merged)
    '''

    # Get Splits to Normalize Train/Val and Test separately
    X_train_val, X_test = train_test_split([i for i in os.listdir(img_path) if i.endswith('.tif')], test_size=0.20,
                                           random_state=42)
    ds = gdal.Open(os.path.join(img_path, X_train_val[0]))
    geotransform = ds.GetGeoTransform()

    array = ds.ReadAsArray()
    array[np.isnan(array)] = 0
    assert not np.any(np.isnan(array)), "Float"

    # Clipping
    array = np.clip(array, a_min=clipping_values[0], a_max=clipping_values[1])

    assert not np.any(np.isnan(array)), "After clipping"
    # Normalize the array

    if standardize:
        # Calculate Means and Standard Deviations per Band
        if 'Sentinel' in img_path:
            if os.path.exists('./s2_means_stds.csv'):
                means_stds_df = pd.read_csv('./s2_means_stds.csv', )

                if ur + '_train_val_means' in means_stds_df.columns:
                    print('Mean and Standard Deviation Values found:')
                    train_val_means = means_stds_df[ur + '_train_val_means'].to_numpy()
                    train_val_stds = means_stds_df[ur + '_train_val_stds'].to_numpy()

                    test_means = means_stds_df[ur + '_test_means'].to_numpy()
                    test_stds = means_stds_df[ur + '_test_stds'].to_numpy()
                    print(train_val_means.shape, train_val_stds.shape, test_means.shape, test_stds.shape)
                else:
                    train_val_means = calc_mean(img_path, X_train_val, input_height, input_width, clipping_values, channels)
                    train_val_stds = calc_std(train_val_means, img_path, X_train_val, input_height, input_width,
                                              clipping_values, channels)

                    test_means = calc_mean(img_path, X_test, input_height, input_width, clipping_values, channels)
                    test_stds = calc_std(train_val_means, img_path, X_test, input_height, input_width, clipping_values,
                                         channels)

                    means_stds_df[ur + '_train_val_means'] = train_val_means
                    means_stds_df[ur + '_train_val_stds'] = train_val_stds
                    means_stds_df[ur + '_test_means'] = test_means
                    means_stds_df[ur + '_test_stds'] = test_stds
                    means_stds_df.to_csv('./s2_means_stds.csv')
            else:
                train_val_means = calc_mean(img_path, X_train_val, input_height, input_width, clipping_values, channels)
                train_val_stds = calc_std(train_val_means, img_path, X_train_val, input_height, input_width,
                                          clipping_values, channels)

                test_means = calc_mean(img_path, X_test, input_height, input_width, clipping_values, channels)
                test_stds = calc_std(train_val_means, img_path, X_test, input_height, input_width, clipping_values,
                                     channels)
                means_stds_df = pd.DataFrame(list(zip(train_val_means, train_val_stds, test_means, test_stds)),
                                             columns=[ur + '_train_val_means', ur + '_train_val_stds', ur + '_test_means',
                                                      ur + '_test_stds'])
                means_stds_df.to_csv('./s2_means_stds.csv')
        elif 'VIIRS' in img_path:
            if os.path.exists('./viirs_means_stds.csv'):
                means_stds_df = pd.read_csv('./viirs_means_stds.csv', )

                if ur + '_' + '_train_val_means' in means_stds_df.columns:
                    print('Mean and Standard Deviation Values found:')
                    train_val_means = means_stds_df[ur + '_' + year + '_train_val_means'].to_numpy()
                    train_val_stds = means_stds_df[ur + '_' + year + '_train_val_stds'].to_numpy()

                    test_means = means_stds_df[ur + '_' + year + '_test_means'].to_numpy()
                    test_stds = means_stds_df[ur + '_' + year + '_test_stds'].to_numpy()
                    print(train_val_means.shape, train_val_stds.shape, test_means.shape, test_stds.shape)
                else:
                    train_val_means = calc_mean(img_path, X_train_val, input_height, input_width, clipping_values, channels)
                    train_val_stds = calc_std(train_val_means, img_path, X_train_val, input_height, input_width,
                                              clipping_values, channels)

                    test_means = calc_mean(img_path, X_test, input_height, input_width, clipping_values, channels)
                    test_stds = calc_std(test_means, img_path, X_test, input_height, input_width, clipping_values, channels)

                    means_stds_df[ur + '_' + year + '_train_val_means'] = train_val_means
                    means_stds_df[ur + '_' + year + '_train_val_stds'] = train_val_stds
                    means_stds_df[ur + '_' + year + '_test_means'] = test_means
                    means_stds_df[ur + '_' + year + '_test_stds'] = test_stds
                    means_stds_df.to_csv('./viirs_means_stds.csv')
            else:
                train_val_means = calc_mean(img_path, X_train_val, input_height, input_width, clipping_values, channels)
                train_val_stds = calc_std(train_val_means, img_path, X_train_val, input_height, input_width,
                                          clipping_values, channels)

                test_means = calc_mean(img_path, X_test, input_height, input_width, clipping_values, channels)
                test_stds = calc_std(test_means, img_path, X_test, input_height, input_width, clipping_values, channels)
                means_stds_df = pd.DataFrame(list(zip(train_val_means, train_val_stds, test_means, test_stds)),
                                             columns=[ur + '_' + year + '_train_val_means',
                                                      ur + '_' + year + '_train_val_stds',
                                                      ur + '_' + year + '_test_means',
                                                      ur + '_' + year + '_test_stds'])
                means_stds_df.to_csv('./viirs_means_stds.csv')

    # Normalize and Resize Images and save the preprocessed Images for Training
    for image in tqdm(X_train_val):
        standardize_resize(img=image, img_path=img_path, input_height=input_height, input_width=input_width,
                           clipping_values=clipping_values, means=train_val_means, stds=train_val_stds,
                           standardize=True)

    for image in tqdm(X_test):
        standardize_resize(img=image, img_path=img_path, input_height=input_height, input_width=input_width,
                           clipping_values=clipping_values, means=test_means, stds=test_stds,
                           standardize=True)


if __name__ == '__main__':

    main(
        img_path='/mnt/datadisk/data/VIIRS/preprocessed/asset/urban/mozambique_2016_2021',
        ur='u',
        year='all',
        input_height=200,
        input_width=200,
        clipping_values=clipping_values,
        channels=[],
        standardize = True
    )
