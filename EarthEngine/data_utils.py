import os
import numpy as np

import rasterio
import pandas as pd
import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
#from tqdm import tqdm

AUTOTUNE = tf.data.AUTOTUNE


# Calculate mean for each channel over all pixels for training set; for validation and test set you need to take
# mean and std of training set as well as in real case scenarios you don't know them beforehand to calculate them
def calc_mean(img_dir: str, img_list: list, input_height: int, input_width: int, clipping_values: list, channels: list):
    '''Calculate mean pixel values per channel over all input images
    
    Args:
        img_dir (str): Path to image data
        input_height (int): Pixel height of input
        input_width (int): Pixel width of input
        clipping_values (list): Interval of min and max values for clipping
        channels (list):  Channels to use; [] if all channels are to be used

    Returns (np.array): Means of pixel values per channel
    '''

    # Ensures that only tif data are in the current directory, if not it will throw an error
    assert all([i.endswith('.tif') for i in img_list])

    # Variable to save the summation of the pixels values
    sum_arr = 0
    # Count of pixels
    sum_pixel = input_height * input_width * len(img_list)

    for img_name in tqdm(img_list):
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
    '''Calculate standard deviation values per channel over all input images
    
    Args:
        means (np.array): Result of calc_mean: Mean of pixel values for each channel
        img_dir (str): Path to image data
        input_height (int): Pixel height of input
        input_width (int): Pixel width of input
        clipping_values (list): Interval of min and max values for clipping
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
    for img_name in tqdm(img_list):
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

def create_splits(img_dir: str, pre2015_path: str, wealth_path: str, urban_rural: str, subset=False):
    '''Create train/val and testsplit for cross validation.
    
    Args:
        img_dir (str):      Path to image data
        pre2015_path(str):  Path to test images with corresponding label dated older than 2015
        wealth_path (str):  Path to label csv files
        urban_rural (str):  One of ['u','r','ur'] to choose whether to use only urban/rural clusters or all data
        subset (bool):      Whether or not to use a subset (for testing)

    Returns:
        X_train_val (list):         List containing filenames for train and validation split
        X_test (list):              List containing filenames for test split
        y_train_val (np.ndarray):   Numpy array containing Asset Wealth (label data) for train and validation split
        y_test (np.ndarray):        Numpy array containing Asset Wealth (label data) for test split

        If pre2015_path is set also returns:
        X_test_pre2015 (list):      List containing filenames for test split with corresponding label dated older than 2015
        y_test_pre2015 (np.ndarray):Numpy array containing Asset Wealth (label data) for test split (dated older than 2015)
    '''

    # Either create a subset of the data to check for possible bugs and errors
    if subset:
        img_list = [img for img in os.listdir(img_dir) if img.endswith('.tif')][:50]

        if urban_rural == 'u' or urban_rural == 'U':
            assert all([i.endswith('u_2.0.tif') for i in img_list])
        elif urban_rural == 'r' or urban_rural == 'R':
            assert all([i.endswith('r_10.0.tif') for i in img_list])
    # Or run code with all selected data
    else:
        img_list = [img for img in os.listdir(img_dir) if img.endswith('.tif')]

        if urban_rural == 'u' or urban_rural == 'U':
            assert all([i.endswith('u_2.0.tif') for i in img_list])
        elif urban_rural == 'r' or urban_rural == 'R':
            assert all([i.endswith('r_10.0.tif') for i in img_list])

    # Gather Label Data
    wealth_df = combine_wealth_dfs(wealth_path)
    y = []
    print('Gathering Label Data')
    # Get corresponding Label for each Image
    for img in tqdm(img_list):
        y.append(get_label_for_img(wealth_df, img).WEALTH_INDEX.iloc[0])
    y = np.array(y)

    # Ensure that y contains a Label for each Image
    assert len(img_list) == len(y)

    # Split data into Train/Validation and Testset
    X_train_val, X_test, y_train_val, y_test = train_test_split(img_list, y, test_size=0.20, random_state=42)

    # If a second evaluation is to be performed on data older than 2015
    # create a second test set
    if pre2015_path:
        X_test_pre2015 = [img for img in os.listdir(pre2015_path) if img.endswith('.tif')]
        y_test_pre2015 = []
        for img in tqdm(X_test_pre2015):
            y_test_pre2015.append(get_label_for_img(wealth_df, img).WEALTH_INDEX.iloc[0])

        return X_train_val, X_test, y_train_val, y_test, X_test_pre2015, y_test_pre2015
    else:
        return X_train_val, X_test, y_train_val, y_test


def generator(img_dir: str, X: list, y: np.ndarray, batch_size: int, input_height: int, input_width: int,
              channel_size: int):
    '''Data generator to generate label and feature batches.
    
    Args:
        img_dir (str):      Path to image data
        X (list):           List containing filenames of split
        y (np.ndarray):     Array containing lbel values of split
        batch_size (int):   Size of training batches
        input_height (int): Pixel height of input
        input_width (int):  Pixel width of input
        channels (int):     Number of channels

    Returns
        batch_x (np.ndarray): Feature batch
        batch_y (np.ndarray): Label batch
    '''
    assert all([i.endswith('.tif') for i in X])

    # generate batches (x : input, y: label)
    batch_x = np.zeros(shape=(batch_size, channel_size, input_height, input_width))
    batch_y = np.zeros(shape=(batch_size,), dtype=float)

    # Iterator
    batch_ele = 0
    for index, img in enumerate(X):
        # Read in each Image
        with rasterio.open(os.path.join(img_dir, img)) as i:
            array = i.read().astype("float32")

        # Ensure that the Array is not empty
        array[np.isnan(array)] = 0
        assert not np.any(np.isnan(array)), "Float"

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
    '''Combines all label csv files to one.
    
    Args:
        wealth_csv_path (str): Path to label csv files

    Returns:
        complete_wealth_df (pd.DataFrame): Pandas DataFrame containing all clusters
    '''

    # Get all Wealth CSV Files
    wealth_files = glob.glob(wealth_csv_path + "/*.csv")
    wealth_df_list = []

    #Combine all Files to one DataFrame
    for filename in wealth_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        wealth_df_list.append(df)

    complete_wealth_df = pd.concat(wealth_df_list, axis=0, ignore_index=True)
    return complete_wealth_df


def get_label_for_img(wealth_df: pd.DataFrame, img_filename: str):
    '''Get label data for a cluster based on the filename.
    
    Args:
        wealth_df (pd.DataFrame):   Pandas DataFrame containing all clusters
        img_dir:                    Path to image data

    Returns:
        wealth_sentinel_df:         Pandas DataFrame including the Asset Wealth Value of the image
    '''
    #
    img_info_df = pd.DataFrame([list(get_img_coordinates(img_filename)) + [img_filename]],
                               columns=['LATNUM', 'LONGNUM', 'Filename'])

    wealth_df['LATNUM'] = wealth_df.LATNUM.apply(lambda x: truncate(x, 4))
    wealth_df['LONGNUM'] = wealth_df.LONGNUM.apply(lambda x: truncate(x, 4))
    label = wealth_df.merge(img_info_df, on=['LATNUM', 'LONGNUM'])[['WEALTH_INDEX', 'SURVEY_YEAR', 'LATNUM','LONGNUM']]

    return label


def get_img_coordinates(img: str):
    '''Extract the cluster coordinates from a given filename.
    
    Args:
        img (str): Filename of image

    Returns:
        str, str : Latitude, longitude

    '''
    return img.split('_')[0], img.split('_')[1]


def truncate(f, n):
    '''Truncates a float f to n decimal places without rounding.
    
    Args:
        f: Float value
        n: Number of decimal places
    '''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d + '0' * n)[:n]])


def get_mean(wealth_df:pd.DataFrame):
    '''Calculate the mean value for WEALTH_INDEX column of a Pandas DataFrame.

    Args:
        wealth_df: Pandas DataFrame containing at least a column 'WEALTH_INDEX'

    Returns:
        float: Mean Asset Wealth of DataFrame
    '''
    return wealth_df.WEALTH_INDEX.mean()
def get_median(wealth_df:pd.DataFrame):
    '''Calculate the Median Value for WEALTH_INDEX column of a Pandas DataFrame.

    Args:
        wealth_df: Pandas Dataframe containing at least a column 'WEALTH_INDEX'

    Returns:
        float: Median Asset Wealth of DataFrame
    '''
    return wealth_df.WEALTH_INDEX.median()
def get_std(wealth_df:pd.DataFrame):
    '''Calculate the Standard Deviation for WEALTH_INDEX column of a Pandas DataFrame.

    Args:
        wealth_df: Pandas DataFrame containing at least a column 'WEALTH_INDEX'

    Returns:
        float: Mean Asset Wealth of DataFrame
    '''
    return wealth_df.WEALTH_INDEX.std()
def get_var(wealth_df:pd.DataFrame):
    '''Calculate the Variance for WEALTH_INDEX column of a Pandas DataFrame.

    Args:
        wealth_df: Pandas DataFrame containing at least a column 'WEALTH_INDEX'

    Returns:
        float: Mean Asset Wealth of DataFrame
    '''
    return wealth_df.WEALTH_INDEX.var()
def get_skew(wealth_df:pd.DataFrame):
    '''Calculate the Skewness for WEALTH_INDEX column of a Pandas DataFrame.

    Args:
        wealth_df: Pandas DataFrame containing at least a column 'WEALTH_INDEX'

    Returns:
        float: Mean Asset Wealth of DataFrame
    '''
    return wealth_df.WEALTH_INDEX.skew()
def get_kurtosis(wealth_df:pd.DataFrame):
    '''Calculate the Kurtosis for WEALTH_INDEX column of a Pandas DataFrame.

    Args:
        wealth_df: Pandas DataFrame containing at least a column 'WEALTH_INDEX'

    Returns:
        float: Mean Asset Wealth of DataFrame

    '''
    return wealth_df.WEALTH_INDEX.kurtosis()


def get_statistics(csv_path: str, timespan_a: list, countries: list, timespan_b=False,
                   timespan_c=False):
    '''Creates a dictionary that includes statistic values per country year and combined per timespan.
    
    The dictionary has the following structure:
    statistics = {
    'country_year': [],
    'mean': [],
    'median': [],
    'std': [],
    'var': [],
    'skewness': [],
    'kurtosis': []
    }
    Args:
        csv_path (str):         Path to label csv files
        timespan_a (list):      Timespan in years e.g. [2012,2013,2014] to include
        countries (list):       Countries to include
        timespan_b (bool/list): Optional: Second timespan in years e.g. [2015] to include
        timespan_c (bool/list): Optional: Third timespan in years e.g. [2016, 2017,2018,2019,2020] to include

    Returns:
        statistics (dict):      Dictionary including statistic values per country year and combined over timespan(s)
    '''
    statistics = {
        'country_year': [],
        'mean': [],
        'median': [],
        'std': [],
        'var': [],
        'skewness': [],
        'kurtosis': []
    }
    for t in [timespan_a, timespan_b, timespan_c]:
        if t:
            # Get Asset Wealth
            wealth_df = combine_wealth_dfs(csv_path)
            # Only use chosen countries and group data
            wealth_df = wealth_df[wealth_df.COUNTRY.isin(countries)]
            grouped = wealth_df.groupby(['COUNTRY'])

            for group_name in grouped.groups.keys():
                group = grouped.get_group(group_name)
                # Only use data for corresponding timespan t
                group = group[group.SURVEY_YEAR.isin(t)]
                if not group.empty:
                    statistics['country_year'].append(group.DHSCC.iloc[1] + '_' + str(group.SURVEY_YEAR.iloc[1]))
                    statistics['mean'].append(get_mean(group))
                    statistics['median'].append(get_median(group))
                    statistics['std'].append(get_std(group))
                    statistics['var'].append(get_var(group))
                    statistics['skewness'].append(get_skew(group))
                    statistics['kurtosis'].append(get_kurtosis(group))
            # Get combined statistics for each timespan
            wealth_df = wealth_df[wealth_df.SURVEY_YEAR.isin(t)]
            statistics['country_year'].append('kombiniert_' + str(t[0]) + '_' + str(t[-1]))
            statistics['mean'].append(get_mean(wealth_df))
            statistics['median'].append(get_median(wealth_df))
            statistics['std'].append(get_std(wealth_df))
            statistics['var'].append(get_var(wealth_df))
            statistics['skewness'].append(get_skew(wealth_df))
            statistics['kurtosis'].append(get_kurtosis(wealth_df))
        else:
            continue

    return statistics

def get_ur_statistics(csv_path:str, timespan_a:list, countries: list, timespan_b=False, timespan_c=False):
    '''Creates a dictionary that includes statistic per region type (urban/rural) per timespan.
    The dictionary has the following keys:
    statistics = { 
    'year': [], 
    'ur': [], 
    'mean': [],
    'median': [],
    'std': [],
    'var': [],
    'skewness': [],
    'kurtosis': []
    }
    
    Args:
        csv_path (str):         Path to label csv files
        timespan_a (list):      Timespan in years e.g. [2012,2013,2014] to include
        countries (list):       Countries to include
        timespan_b (bool/list): Optional: Second timespan in Years e.g. [2015] to include
        timespan_c (bool/list): Optional: Third timespan in Years e.g. [2016, 2017,2018,2019,2020] to include

    Returns:
        statistics (dict):      Dictionary including statistic values per region type (urban/rural) per timespan.
    '''
    statistics = {
                'year': [],
                'ur': [],
                'mean': [],
                'median': [],
                'std': [],
                'var': [],
                'skewness': [],
                'kurtosis': []
            }
    for t in [timespan_a, timespan_b, timespan_c]:
        if t:
            wealth_df = combine_wealth_dfs(csv_path)
            wealth_df = wealth_df[wealth_df.COUNTRY.isin(countries)]
            wealth_df = wealth_df[wealth_df.SURVEY_YEAR.isin(t)]
            group_ur = wealth_df.groupby(['URBAN_RURA'])
            for ur in group_ur.groups.keys():
                group = group_ur.get_group(ur)
                statistics['year'].append(str(t[0])+'-'+str(t[-1]))
                if ur == 0:
                    statistics['ur'].append('urban')
                else:
                    statistics['ur'].append('rural')
                statistics['mean'].append(get_mean(group))
                statistics['median'].append(get_median(group))
                statistics['std'].append(get_std(group))
                statistics['var'].append(get_var(group))
                statistics['skewness'].append(get_skew(group))
                statistics['kurtosis'].append(get_kurtosis(group))
    return statistics
