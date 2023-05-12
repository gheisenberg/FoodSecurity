import rasterio
import tensorflow as tf
import warnings
import os
import math
import numpy as np

#own imports


def read_geotiff_float(file, channel_l=False):
    if isinstance(file, tf.Tensor):
        file = bytes.decode(file.numpy())
    with rasterio.open(file) as img:
        if not channel_l:
            array = img.read().astype("float32")
        else:
            array = img.read(channel_l).astype("float32")
        profile = img.profile
    return array, profile


def crop_mid_array(array, height, width):
    """crops the array to a fixed height, width while ensuring to crop from all sides so that the center point
    stays in the center
    Caution: The georeferencing is not adjusted - it is just preprocessing! The GeoTiff will not be in the correct
    position if visualized! To correct this look at, e.g.: https://github.com/rasterio/affine, or better GeoPandas, e.g:
    https://automating-gis-processes.github.io/CSC18/lessons/L6/clipping-raster.html
    """
    shape = array.shape
    h_start = math.floor((shape[1] - height)/2.)
    w_start = math.floor((shape[2] - width)/2.)
    if h_start < 0 or w_start < 0:
        raise ValueError(f"Wrong dimensions for cropping: In Array = {shape} expected min: {height, width}")
    array = array[:,h_start:h_start + height, w_start:w_start + width]
    return array


def fill_NaNs(array, replace_nan_value='local channel mean', null_sum=True):
    """Replaces NaNs with a specified value, a specified value per channel or the channel mean

    Args:
        array: np.array
        replace_nan_value: the value to replace NaNs with. May be a number, a 1D np.array,
        or 'local channel mean' (uses mean of the channel) (recommended).
    """
    #replacing nans
    nulls = 0
    if np.isnan(array).any():
        if null_sum:
            nulls = np.isnan(array).sum()/ (array.shape[0]*array.shape[1]*array.shape[2]) * 100
        if type(replace_nan_value) == int or type(replace_nan_value) == float:
            array = np.nan_to_num(array, replace_nan_value)
        elif type(replace_nan_value) == np.array or replace_nan_value == 'local channel mean':
            if replace_nan_value == 'local channel mean':
                replace_nan_value = np.nanmean(array, axis=(1,2))
            #actual replacing
            array = np.array([np.nan_to_num(array_i, nan=rep_i) for array_i, rep_i in zip(array, replace_nan_value)])
        else:
            raise NotImplementedError()
    return array, nulls


def clip_minmax_values(array, clipping_values='outlier', std_multiplier=3):
    """Clips values at min and max bounds

    Args:
        clipping_values: Minimum and maximum values for the array. Recommended: Use 'outlier' in combination with the
        std_multiplier for outlier detection with mean +- std_multiplier * std. Recommended
        std_multiplier is 3. Otherwise white areas might be clipped. You can also use tuples (min_v, max_v) to clip
        at hard values.
        std_multiplier: see clipping values"""
    if type(clipping_values) == tuple:
        array = np.clip(array, a_min = clipping_values[0], a_max = clipping_values[1])
    elif clipping_values == 'outlier':
        means = array.mean(axis=(1, 2))
        stds = array.std(axis=(1, 2))
        if std_multiplier is False or means is False or stds is False:
            raise ValueError(f"Missing some values: means {means}, stds {stds} or std_multiplier {std_multiplier}")
        else:
            min_arr = means - std_multiplier * stds
            max_arr = means + std_multiplier * stds
            array = np.array([np.clip(array_i, a_min=mini, a_max=maxi) for array_i, mini, maxi in zip(array, min_arr,
                                                                                                      max_arr)])
    elif not clipping_values:
        pass
    else:
        raise NotImplementedError()
    return array


def load_geotiff(file, height, width, channel_l=False, replace_nan_value=False, clipping_values=False, std_multiplier=False,
                 drop_perc_NaNs=False, delete_perc_NaNs=False, only_return_array=False):
    """Loads geotiffs with specified height, width, channels. Also replaces NaN values and clips outliers if desired."""
    array, profile = read_geotiff_float(file, channel_l)
    if height and width:
        array = crop_mid_array(array, height, width)
    if height and not width or not height and width:
        raise  NotImplementedError("Specify height and width")
    if replace_nan_value:
        array, nulls = fill_NaNs(array, replace_nan_value)
    if clipping_values:
        array = clip_minmax_values(array, clipping_values, std_multiplier)
    if drop_perc_NaNs and drop_perc_NaNs <= nulls:
        warnings.warn(f'Droping file because of missing values = {nulls} % {file}')
        array = False
    if delete_perc_NaNs and delete_perc_NaNs <= nulls:
        warnings.warn(f'Deleting file because of missing values = {nulls} %')
        os.remove(file)
    if only_return_array:
        return array
    else:
        return array, nulls, profile


