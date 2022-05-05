import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from osgeo import osr
from osgeo import gdal
from data_utils import calc_mean, calc_std


def pad_to_input_size(array: np.ndarray, input_height: int, input_width: int):
    '''
    Pads Images to uniform Size
    Args:
        array (np.ndarray): Image Array
        input_height (int): Uniform Image Height (px)
        input_width (int):  Uniform Image Width (px)

    Returns:
        array (np.ndarray): Padded Array (np.ndarray)
    '''
    # S2 Images:
    if len(array.shape) == 3:
        padding_y = int((input_height - array.shape[1]) / 2)
        padding_x = int((input_width - array.shape[2]) / 2)

        calculated_height = padding_y * 2 + array.shape[1]
        calculated_width = padding_x * 2 + array.shape[2]
        if calculated_width == input_width and calculated_height == input_height:
            array = np.pad(array, ((0, 0), (padding_y, padding_y), (padding_x, padding_x)), constant_values=0)
        elif calculated_width != input_width and calculated_height != input_height:
            array = np.pad(array, ((0, 0), (padding_y + 1, padding_y), (padding_x + 1, padding_x)), constant_values=0)
        elif calculated_width != input_width:
            array = np.pad(array, ((0, 0), (padding_y, padding_y), (padding_x + 1, padding_x)), constant_values=0)

        elif calculated_height != input_height:
            array = np.pad(array, ((0, 0), (padding_y + 1, padding_y), (padding_x, padding_x)), constant_values=0)

    # VIIRS Images:
    elif len(array.shape) == 2:
        padding_y = int((input_height - array.shape[0]) / 2)
        padding_x = int((input_width - array.shape[1]) / 2)

        calculated_height = padding_y * 2 + array.shape[0]
        calculated_width = padding_x * 2 + array.shape[1]
        if calculated_width == input_width and calculated_height == input_height:
            array = np.pad(array, ((padding_y, padding_y), (padding_x, padding_x)), constant_values=0)
        elif calculated_width != input_width and calculated_height != input_height:
            array = np.pad(array, ((padding_y + 1, padding_y), (padding_x + 1, padding_x)), constant_values=0)
        elif calculated_width != input_width:
            array = np.pad(array, ((padding_y, padding_y), (padding_x + 1, padding_x)), constant_values=0)

        elif calculated_height != input_height:
            array = np.pad(array, ((padding_y + 1, padding_y), (padding_x, padding_x)), constant_values=0)

    return array


def slice_to_input_size(array: np.ndarray, input_height: int, input_width: int):
    '''
    Slice image to uniform size.
    Args:
        array (np.ndarray):         Image array
        input_height (int):         Uniform Image Height (px)
        input_width (int):          Uniform Image Width (px)

    Returns:
        array (np.ndarray): Sliced Image Array

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


def standardize_resize(img: str, img_path: str, input_height: int, input_width: int, clipping_values: list,
                       means: np.ndarray, stds: np.ndarray, standardize: bool):
    '''
    Standardize and Resize a GeoTIFF.
    Args:
        img (str):              Filename
        img_path (str):         Path to Directory of GeoTIFF
        input_height (int):     Uniform Image Height (px)
        input_width (int):      Uniform Image Width (px)
        clipping_values (list): Interval of Min and Max Values for Clipping
        means (np.ndarray):     Mean values per band
        stds (np.ndarray):      Standard deviation per band
        standardize (bool):     If True: Image is standardized and resized to input size
                                If False: Image is only resized to input size

    Returns:

    '''
    ds = gdal.Open(os.path.join(img_path, img))
    geotransform = ds.GetGeoTransform()
    array = ds.ReadAsArray()

    array[np.isnan(array)] = 0
    assert not np.any(np.isnan(array)), "Float"

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
            if array.shape[1] < input_height:
                array = pad_to_input_size(array, input_height, input_width)
            else:
                array = slice_to_input_size(array, input_height, input_width)

        assert array.shape == (array.shape[0], input_height, input_width)
    elif len(array.shape) == 2:
        if array.shape != (input_height, input_width):
            if array.shape[0] < input_height:
                array = pad_to_input_size(array, input_height, input_width)
            else:
                array = slice_to_input_size(array, input_height, input_width)

        assert array.shape == (input_height, input_width)
    # return normalized and resized image array

    # Create new geotiff
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

    # For VIIRS Data: copy the band values to two more bands to create the illusion of a RGB Image
    # for better compatibility with CNN
    elif 'VIIRS' in img_path:
        array = np.concatenate((np.array([array]),) * 3, axis=0)

        for i in range(len(array)):
            dst_ds.GetRasterBand(i + 1).WriteArray(array[i])  # write band to the raster
    dst_ds.FlushCache()  # write to disk
    dst_ds = None


def main(img_path: str, ur: str, input_height: str, input_width: str, clipping_values: list, channels: list):
    '''

    Args:
        img_path (str):             Path to Image Data
        ur (str):                   'u' for Urban, 'r' for Rural, 'ur' for all data
        input_height (int):         Uniform Image Height (px)
        input_width (int):          Uniform Image Width (px)
        clipping_values (list):     Interval of Min and Max Values for Clipping
        channels (list): List of    Channels to use. [] to use all Channels.

    Returns:

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

    # Calculate Means and Standard Deviations per Band
    # Mean and Standard Deviation Values are stored in a csv file so the calculation only has to be performed once
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
                means_stds_df.to_csv('./viirs_means_stds.csv')
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
            means_stds_df.to_csv('./viirs_means_stds.csv')

    # # Normalize and Resize Images and save the preprocessed Images for Training
    for image in tqdm(X_train_val):
        standardize_resize(image, img_path, input_height, input_width, clipping_values, train_val_means, train_val_stds)

    for image in tqdm(X_test):
        standardize_resize(image, img_path, input_height, input_width, clipping_values, test_means, test_stds)


if __name__ == '__main__':

    main(
        img_path='/mnt/datadisk/data/VIIRS/preprocessed/asset/urban_rural/',
        ur='ur',
        input_height=1340,
        input_width=1340,
        clipping_values=[0, 3000],
        channels=[]
    )