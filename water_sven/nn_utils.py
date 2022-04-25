import os
from sklearn.utils import class_weight
import time
import rasterio
import numpy as np
import random
import warnings


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
    train_img = os.listdir(train_path)
    label_array = np.zeros(len(train_img), dtype=int)
    index = 0

    #Create array with the labels used (training labels) such that we get a list of the form [1 1 1 0 2 1 2....]
    #necessary: we need to ensure that we get a label for every image, otherwise it needs to be excluded!
    for img in train_img:
        got = False
        for pos, survey_name in enumerate(labels_df['name']):
            if survey_name in img:
                label = labels_df.loc[pos]['label']
                label_array[index] = label
                index += 1
                got = True
        if not got:
            warnings.warn("Could not find label for train image" + img)

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(label_array), y=label_array)
    #looks suspicious, should be alright though (works at least)
    class_weights_d = dict(enumerate(class_weights,))
    return class_weights_d


#Generate iterable object for model.fit()
def generator(x_path, labels, batch_size, input_height, input_width, clipping_values, channels,
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
    # print(x_list)
    # for x in x_list:
    #     print(x)
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
        if normalize:
            array = ((array.transpose(1,2,0)-means)/stds).transpose(2, 0, 1)
            assert not np.any(np.isnan(array)), "Normalize"
        # Add to batch
        batch_x[batch_ele] = array
        #print('Array', type(array), '\n', array)
        #array = cv2.resize(array, (224,224))     # resize image to match model's expected sizing
        #array = array.reshape(1,224,224,3) # return the image with shaping that TF wants.
        # print('Array', type(array), '\n', array)
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
