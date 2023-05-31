import os

import pandas as pd
from sklearn.utils import class_weight
import time
import rasterio
import numpy as np
import random
import warnings
import helper_utils as hu
import config as cfg
import tensorflow as tf
import math
from functools import partial

###private imports
import geo_utils as gu
logger = hu.setup_logger(cfg.logging)

#Create class weights for the prediction model as our data set is imbalanced (acoount higher weight to classes with
#less samples)
def create_class_weights_dict(train_path, labels_df):
    """Creates class_weights with sklearn.utils.class_weight.compute_class_weight method

    Args:
        train_path (str): Path where train images are stored
        labels_df (Pandas DF): DataFrame with label data

    Returns:
        class_weights_d (dict): Weights for every class
    """
    # train_img = os.listdir(train_path)
    # label_array = np.zeros(len(train_img), dtype=int)
    # index = 0

    #Create array with the labels used (training labels) such that we get a list of the form [1 1 1 0 2 1 2....]
    #necessary: we need to ensure that we get a label for every image, otherwise it needs to be excluded! Deprecated -
    #Deprecated: only labels with files are available now
    # for img in train_img:
    #     got = False
    #     for pos, survey_name in enumerate(labels_df['name']):
    #         if survey_name in img:
    #             label = labels_df.loc[pos]['label']
    #             label_array[index] = label
    #             index += 1
    #             got = True
    #     if not got:
    #         warnings.warn("Could not find label for train image" + img)

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels_df['label']), y=labels_df['label'])
    #looks suspicious, should be alright though (works at least)
    class_weights_d = dict(enumerate(class_weights,))
    return class_weights_d


def dataset_creator(df, height, width, channel_l, shuffle=False, batch_size=16, cache_p=False, verbose=False, prediction_type='regression', num_labels=False):
    steps_per_epoch = math.ceil(len(df)/batch_size)
    files = tf.data.Dataset.from_tensor_slices(tf.constant(df['path']))
    labels = np.array(df['label'])
    if prediction_type == 'regression':
        labels = labels.reshape(-1, 1)
    elif prediction_type == 'categorical':
        labels = tf.constant(labels)
        labels = tf.one_hot(labels, num_labels)
    else:
        raise ValueError(f"cfg.type_m or respectively prediction_type must be 'regression' or 'categorical' it is"
                         f" {prediction_type} though")
    labels = tf.data.Dataset.from_tensor_slices(labels)
    assert len(files) == len(labels)
    if verbose:
        logger.debug('files ds %s', files)
        logger.debug('labels ds %s', labels)
    func = partial(gu.load_geotiff, height=height, width=width, channel_l=channel_l, only_return_array=True)
    ds = files.map(lambda x: tf.py_function(func, [x], tf.float32),
                num_parallel_calls=tf.data.AUTOTUNE)
    if verbose:
        logger.debug('Read imgs ds %s', ds)
    assert len(ds) == len(labels)
    ds = tf.data.Dataset.zip((ds, labels))
    if verbose:
        logger.debug('Zipped ds %s %s', ds, len(ds))
    #significantly improves speed! ~25%
    if cache_p:
        ds = ds.cache(cache_p)
        #buffer_size does not seem to influence the performance
        #to do: set on True again?!
    if shuffle:
        ds = ds.shuffle(batch_size, reshuffle_each_iteration=False)
    ds = ds.batch(batch_size)
    #does not seem to have an influence when used with cached datasets
    ds = ds.prefetch(tf.data.AUTOTUNE)
    if verbose:
        logger.debug('Final ds %s', ds)
        logger.debug('steps p epoch %s', steps_per_epoch)
    return ds, steps_per_epoch


def dataset_creator_production(df, height, width, channel_l, shuffle=False, batch_size=16, cache_p=False, verbose=False, prediction_type='regression', num_labels=False):
    steps_per_epoch = math.ceil(len(df)/batch_size)
    files = tf.data.Dataset.from_tensor_slices(tf.constant(df['path']))
    if verbose:
        logger.debug('files ds %s', files)
    func = partial(gu.load_geotiff, height=height, width=width, channel_l=channel_l, only_return_array=True)
    ds = files.map(lambda x: tf.py_function(func, [x], tf.float32),
                num_parallel_calls=tf.data.AUTOTUNE)
    if verbose:
        logger.debug('Read imgs ds %s', ds)
    #significantly improves speed! ~25%
    if cache_p:
        ds = ds.cache(cache_p)
        #buffer_size does not seem to influence the performance
        #to do: set on True again?!
    if shuffle:
        ds = ds.shuffle(batch_size, reshuffle_each_iteration=False)
    ds = ds.batch(batch_size)
    #does not seem to have an influence when used with cached datasets
    ds = ds.prefetch(tf.data.AUTOTUNE)
    if verbose:
        logger.debug('Final ds %s', ds)
        logger.debug('steps p epoch %s', steps_per_epoch)
    return ds, steps_per_epoch


#Generate iterable object for model.fit()
def generator_old(x_path, labels, batch_size, input_height, input_width, clipping_values, channels,
              channel_size, num_labels, normalize=False):
    """from Shannon
        will be legacy code soon - thus no detailed description! Typical generator though!
        Input:
        channels - list
        steps:
        1. list all files
        2. shuffle list
        3. creates empty arrays
        4. loads w rasterio returns numpy array?!
        5. sets NAN to 0???
        6. np.clip to 0/3000 ??? replace by 2Sigma?
        7. to do: insert statistics (simple decils?)
        8. cropping from one edge --> needs to be fixed
        9 normalization --> generic func
        10. gets label - one hots it, labels = df?
        11. yields batch"""
    x_list = os.listdir(x_path)
    x_list = [x for x in x_list if x.endswith('.tif')]
    # logger.debug(x_list)
    # for x in x_list:
    #     logger.debug(x)
    assert all([i.endswith('.tif') for i in x_list])
    #Shuffle elements in list, so that batches consists of images of different surveys
    random.shuffle(x_list)
    #generate empty/0 batches (x : input, y: label)
    batch_x = np.zeros(shape=(batch_size, channel_size,input_height, input_width))
    batch_y = np.zeros(shape=(batch_size, num_labels), dtype=int)
    #Iterator
    batch_ele = 0

    for x in x_list:
        #Get training sample x
        img_path = os.path.join(x_path, x)

        with rasterio.open(img_path) as img:
            if len(channels) == 0:
                array = img.read().astype("float32")
                #what does it return? (width, height, channels=13?) for S2
            else:
                #channels is a list
                array = img.read(channels).astype("float32")

        array[np.isnan(array)] = 0
        assert not np.any(np.isnan(array)), "Float"
        #Clipping at values 0 n 3000
        array = np.clip(array,a_min = clipping_values[0],a_max = clipping_values[1])

        assert not np.any(np.isnan(array)), "After clipping"
        #Ensure that that all arrays have the same size via cropping
        array = array[:,:input_height,:input_width]

        #Normalize the array
        # if normalize:
        #     array = ((array.transpose(1,2,0)-means)/stds).transpose(2, 0, 1)
        #     assert not np.any(np.isnan(array)), "Normalize"
        # Add to batch
        batch_x[batch_ele] = array
        #logger.debug('Array', type(array), '\n', array)
        #array = cv2.resize(array, (224,224))     # resize image to match model's expected sizing
        #array = array.reshape(1,224,224,3) # return the image with shaping that TF wants.
        # logger.debug('Array', type(array), '\n', array)
        # sys.exit()

        #Get corresponding label y
        #probably slow #comparison w/ str while looping
        for index, survey_name in enumerate(labels['name']):
            if survey_name in x:
                one_hot = np.zeros(shape = num_labels)
                label_pos = (labels.loc[index]['label'])
                #One hot encoding
                one_hot[label_pos] = 1
                batch_y[batch_ele] = one_hot

        #Check if batch is already full (Note: Index in batch array is from 0...4 hence we need to add +1 to batch_ele)
        if (batch_ele+1) == batch_size:
            batch_x = batch_x.transpose(0,2,3,1)
            #Return of batch_x,batch_y
            yield batch_x.astype(np.float32), batch_y.astype(np.float32)
            #Reset settings -> Start of next batch generation
            batch_ele = 0
            batch_x = np.zeros(shape=(batch_size, channel_size,input_height, input_width))
            batch_y = np.zeros(shape=(batch_size, num_labels), dtype=int)

        else:
            batch_ele += 1


def test_imgs(labels_df, channels):
    nr = 0
    miss = 0
    min_shape = 100000000
    for id, row in labels_df.iterrows():
        # Get training sample x
        img_path = row['path']
        #logger.debug(img_path)
        if not id % 100:
            logger.debug(id)
        if not img_path or type(img_path) is not str and np.isnan(img_path):
            if row['DHSYEAR'] >= 2012:
                warnings.warn(f"\n{row[['DHSID', 'GEID', 'LATNUM', 'LONGNUM', 'PCA w_location_weighting all', 'DHSYEAR', 'path']]} has "
                              f"no path {img_path} {miss}")
                miss += 1
        else:
            with rasterio.open(img_path) as img:
                if len(channels) == 0:
                    array = img.read().astype("float32")
                    # what does it return? (width, height, channels=13?) for S2
                else:
                    # channels is a list
                    array = img.read(channels).astype("float32")
            if array.shape[1] < 1000 or array.shape[2] < 1000:
                nr += 1
                min_s = min(array.shape[1], array.shape[2])
                if min_s < min_shape:
                    min_shape = min_s
                warnings.warn(f'{img_path} is to small: {array.shape} number {nr} minimum shape is {min_shape}')
                # os.remove(img_path)
            if array.shape[1] > 1500 or array.shape[2] > 1500:
                warnings.warn(f'{img_path} is to big: {array.shape}')
                os.remove(img_path)
    logger.debug(f"{miss} missing images and {nr} to small images. Smallest size: {min_shape}")




def generator(labels_df, batch_size, input_height, input_width, clipping_values, channels,
              channel_size, num_labels, normalize=False):
    """from Shannon
        will be legacy code soon - thus no detailed description! Typical generator though!
        Input:
        channels - list
        steps:
        1. list all files
        2. shuffle list
        3. creates empty arrays
        4. loads w rasterio returns numpy array?!
        5. sets NAN to 0???
        6. np.clip to 0/3000 ??? replace by 2Sigma?
        7. to do: insert statistics (simple decils?)
        8. cropping from one edge --> needs to be fixed
        9 normalization --> generic func
        10. gets label - one hots it, labels = df?
        11. yields batch"""
    # x_list = os.listdir(x_path)
    # x_list = [x for x in x_list if x.endswith('.tif')]
    # logger.debug(x_list)
    # for x in x_list:
    #     logger.debug(x)
    # assert all([i.endswith('.tif') for i in x_list])
    #Shuffle elements in list, so that batches consists of images of different surveys
    # random.shuffle(x_list)
    #generate empty/0 batches (x : input, y: label)
    batch_x = np.zeros(shape=(batch_size, channel_size,input_height, input_width))
    if num_labels == 1:
        batch_y = np.zeros(shape=(batch_size, num_labels), dtype=np.float32)
    else:
        batch_y = np.zeros(shape=(batch_size, num_labels), dtype=int)
    #Iterator
    batch_ele = 0

    for id, row in labels_df.iterrows():
        #Get training sample x
        img_path = row['path']

        with rasterio.open(img_path) as img:
            if len(channels) == 0:
                array = img.read().astype("float32")
                #what does it return? (width, height, channels=13?) for S2
            else:
                #channels is a list
                array = img.read(channels).astype("float32")
        # logger.debug(array.shape)
        array[np.isnan(array)] = 0
        assert not np.any(np.isnan(array)), "Float"
        #Clipping at values 0 n 3000
        array = np.clip(array,a_min = clipping_values[0],a_max = clipping_values[1])

        assert not np.any(np.isnan(array)), "After clipping"
        #Ensure that that all arrays have the same size via cropping
        array = array[:,:input_height,:input_width]

        #Normalize the array
        # if normalize:
        #     array = ((array.transpose(1,2,0)-means)/stds).transpose(2, 0, 1)
        #     assert not np.any(np.isnan(array)), "Normalize"
        # Add to batch
        batch_x[batch_ele] = array
        #logger.debug('Array', type(array), '\n', array)
        #array = cv2.resize(array, (224,224))     # resize image to match model's expected sizing
        #array = array.reshape(1,224,224,3) # return the image with shaping that TF wants.
        # logger.debug('Array', type(array), '\n', array)
        # sys.exit()

        #Get corresponding label y
        #probably slow #comparison w/ str while looping
        if num_labels == 1:
            batch_y[batch_ele] = row['label']
        else:
            one_hot = np.zeros(shape=num_labels)
            label_pos = row['label']
            # One hot encoding
            one_hot[label_pos] = 1
            batch_y[batch_ele] = one_hot


        #Check if batch is already full (Note: Index in batch array is from 0...4 hence we need to add +1 to batch_ele)
        if (batch_ele+1) == batch_size:
            batch_x = batch_x.transpose(0,2,3,1)
            #Return of batch_x,batch_y
            # logger.debug(batch_y)
            yield batch_x.astype(np.float32), batch_y.astype(np.float32)
            #Reset settings -> Start of next batch generation
            batch_ele = 0
            batch_x = np.zeros(shape=(batch_size, channel_size,input_height, input_width))
            if num_labels == 1:
                batch_y = np.zeros(shape=(batch_size, num_labels), dtype=np.float32)
            else:
                batch_y = np.zeros(shape=(batch_size, num_labels), dtype=int)

        else:
            batch_ele += 1


def transform_data_for_ImageDataGenerator(datasets_l):
    """Loads all data into memory since the generator used right now is really slow (and soon to be legacy code!)
    Also transforms the datasets into the format used by Keras ImageDataGenerator!
    No detailed documentation since this is messy and soon to be legacy code!"""
    ds_l = []
    t_0 = time.time()
    t_ele = 0
    t_transform = 0
    for ds in datasets_l:
        ds_x = []
        ds_y = []
        for nr, ele in enumerate(ds.__iter__()):
            x = 1
            t_1 = time.time()
            for x in ele[0]:
                ds_x.append(x)
            for y in ele[1]:
                ds_y.append(y)
            t_ele += time.time() - t_1

        t1 = time.time()
        ds_x = np.array(ds_x)
        ds_y = np.array(ds_y)
        ds_l.append((ds_x, ds_y))
        t_transform += time.time() - t1
    return ds_l, t_ele, t_transform, time.time() - t_0


def testing_imgs():
    label_csv = '/mnt/datadisk/data/Projects/water/inputs/water_labels.csv'
    labels_df = pd.read_csv(label_csv)
    base_path = '/mnt/datadisk/data/Sentinel2/raw/'
    if base_path:
        # create pathes
        print(labels_df['path'].count())
        labels_df['path'] = np.NaN
        print(labels_df['path'].count())
        labels_df['path'] = base_path + labels_df['GEID'] + labels_df['DHSID'].str[-8:] + '.tif'
        available_files = hu.files_in_folder(base_path)
        # check if actually available
        labels_df["path"] = labels_df['path'].apply(lambda x: x if x in available_files else np.NaN)
    channels = [4, 3, 2]
    test_imgs(labels_df, channels)

#testing_imgs()
