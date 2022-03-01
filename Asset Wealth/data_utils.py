import numpy as np
import os
import rasterio
import random
import pandas as pd
import glob


# Calculate mean for each channel over all pixels for training set; for validation and test set you need to take
# mean and std of training set as well as in real case scenarios you don't know them beforehand to calculate them
def calc_mean(img_dir:str,input_height:int, input_width:int, clipping_values:list, channels:list):
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

    img_list = os.listdir(img_dir)

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
        array = array[:, :input_height, :input_width]
        # Add up the sum over the input and height for each channel separately
        sum_arr += array.sum(axis=(1, 2))

    # Calculate mean for each channel
    means = sum_arr / sum_pixel

    # return mean per channel(#mean == # channels)
    return means


# Calculate standard deviation (note:mean function has to be executed beforehand as it is required as input)
def calc_std(means, img_dir:str, input_height, input_width, clipping_values, channels):
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
    img_list = os.listdir(img_dir)

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
        array = array[:, :input_height, :input_width]

        array = np.power(array.transpose(1, 2, 0) - means, 2).transpose(2, 0, 1)
        sum_arr += array.sum(axis=(1, 2))

    # Second part of equation
    stds = np.sqrt(sum_arr / sum_pixel)

    return stds


# Generate iterable object for model.fit()
def generator(x_dir:str,  labels:pd.DataFrame, batch_size, means, stds, input_height, input_width, clipping_values, channels,
              channel_size):
    '''

    Args:
        x_dir (str): Path to split directory
        labels (pd.DataFrame): Dataframe containing label data for split
        batch_size (int): Size of training batches
        means (np.array): Result of calc_mean: mean pixel values for each channel
        stds (np.array): Result of calc_stds: Standard deviation of pixel values per channel
        input_height (int): pixel height of input
        input_width (int): pixel width of input
        clipping_values (list): interval of min and max values for clipping
        channels (list):  Channels to use; [] if all channels are to be used
        channel_size (int): Number of channels (3 for RGB, 13 for all channels)

    Returns np.array, np.array: data (x) and label (y) batch

    '''
    x_list = os.listdir(x_dir)

    assert all([i.endswith('.tif') for i in x_list])

    # Shuffle elements in list, so that batches consists of images of different surveys
    random.shuffle(x_list)

    # generate batches (x : input, y: label)
    batch_x = np.zeros(shape=(batch_size, channel_size, input_height, input_width))
    batch_y = np.zeros(shape=(batch_size, ), dtype=float)

    # Iterator
    batch_ele = 0

    for x in x_list:
        # Get training sample x
        x_path = os.path.join(x_dir,x)
        with rasterio.open(x_path) as img:
            # if we want to use all channels
            if len(channels) == 0:
                array = img.read().astype("float32")
            else:
                array = img.read(channels).astype("float32")

        array[np.isnan(array)] = 0
        assert not np.any(np.isnan(array)), "Float"
        # Clipping
        array = np.clip(array, a_min=clipping_values[0], a_max=clipping_values[1])

        assert not np.any(np.isnan(array)), "After clipping"
        # Ensure that that all arrays have the same size via cropping
        array = array[:, :input_height, :input_width]

        # Normalize the array
        array = ((array.transpose(1, 2, 0) - means) / stds).transpose(2, 0, 1)
        assert not np.any(np.isnan(array)), "Normalize"
        # Add to batch
        batch_x[batch_ele] = array

        # Get corresponding label y
        for index, survey_name in enumerate(labels['filename']):
            if survey_name in x:
                label = (labels.loc[index]['WEALTH_INDEX'])
                batch_y[batch_ele] = label

        # Check if batch is already full (Note: Index in batch array is from 0...4 hence we need to add +1 to batch_ele)
        if (batch_ele + 1) == batch_size:
            batch_x = batch_x.transpose(0, 2, 3, 1)
            # Return of batch_x,batch_y
            yield batch_x.astype(np.float32), batch_y.astype(np.float32)
            # Reset settings -> Start of next batch generation
            batch_ele = 0
            batch_x = np.zeros(shape=(batch_size, channel_size, input_height, input_width))
            batch_y = np.zeros(shape=(batch_size, ), dtype=float)

        else:
            batch_ele += 1
    return batch_x, batch_y

def combine_wealth_dfs(wealth_csv_path: str):

    wealth_files = glob.glob(wealth_csv_path + "/*.csv")
    wealth_df_list = []

    for filename in wealth_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        wealth_df_list.append(df)

    complete_wealth_df = pd.concat(wealth_df_list, axis=0, ignore_index=True)
    return complete_wealth_df

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])