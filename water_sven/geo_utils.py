import rasterio
import tensorflow as tf
import warnings
import os
import math
import numpy as np
from sklearn.cluster import DBSCAN

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


def cluster_coordinates(df, eps, x='LONGNUM', y='LATNUM', min_samples=2, reassign_noise=False,
                        assign_noise_to_groups=None):
    """
    Clusters geographic coordinates using the DBSCAN algorithm.

    Args:
        df (pandas.DataFrame): DataFrame containing the coordinates.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        x (str): Column name for longitude. Default is 'LONGNUM'.
        y (str): Column name for latitude. Default is 'LATNUM'.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        reassign_noise (bool): If true, assign independent cluster values to noise points. Default is False.
        assign_noise_to_groups (int): If specified, randomly assign noise points to groups of this size. Default is None.

    Returns:
        pandas.DataFrame: DataFrame with an added 'clustered' column and 'clustered: unique non-clustered' column.
        'clustered' contains the cluster labels assigned by DBSCAN, 'clustered: unique non-clustered' contains unique
        cluster values for noise points and the original clusters if reassign_noise is True, and grouped cluster values
        for noise points if assign_noise_to_groups is specified.
    """
    # Extract longitude and latitude from DataFrame
    coordinates = df[[x, y]].values

    # Convert eps from kilometers to radians for use by haversine
    kms_per_radian = 6371.0088
    eps_radians = eps / kms_per_radian

    # Run DBSCAN
    db = DBSCAN(eps=eps_radians, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(
        np.radians(coordinates))

    # Add cluster labels to DataFrame
    df['clustered'] = db.labels_

    if reassign_noise:
        # Make a copy of cluster labels
        unique_cluster_labels = db.labels_.copy()

        # Find the maximum cluster label (ignoring noise labeled as -1)
        max_cluster_label = max(unique_cluster_labels)

        # Identify noise points
        noise_points = unique_cluster_labels == -1

        # Assign each noise point a unique cluster value, starting from max_cluster_label + 1
        unique_cluster_labels[noise_points] = range(-1, -1 - sum(noise_points), -1)

        df['clustered: unique non-clustered'] = unique_cluster_labels

    if assign_noise_to_groups is not None:
        # Make a copy of cluster labels
        grouped_cluster_labels = db.labels_.copy()

        # Identify noise points
        noise_points = np.where(grouped_cluster_labels == -1)[0]

        # Shuffle the noise points
        np.random.shuffle(noise_points)

        # Assign each group of noise points a unique cluster value
        label = 0
        for i in range(0, len(noise_points), assign_noise_to_groups):
            label -= 1
            grouped_cluster_labels[noise_points[i:i + assign_noise_to_groups]] = label

        df['clustered: grouped non-clustered'] = grouped_cluster_labels

    return df



def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r
