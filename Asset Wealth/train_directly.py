from vvg19 import VVG19_hyperspectral
from data_utils import generator, calc_mean, calc_std

from functools import partial

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras import optimizers

import pandas as pd
import os
import numpy as np


def main(img_dir: str, subset: bool, input_height: int, input_width: int,
         channels: list, channel_size: int, clipping_values: list, batch_size: int):
    '''

    Args:
        img_dir (str): Path to image data
        subset (bool): Whether or not to use a subset to test the process
        input_height (int): pixel height of input
        input_width (int): pixel width of input
        channels (list):  Channels to use; [] if all channels are to be used
        channel_size (int): Number of channels (3 for RGB, 13 for all channels)
        clipping_values (list): interval of min and max values for clipping
        batch_size (int): Size of training batches

    Returns:

    '''

    ## get data and labels
    if subset:
        train_dir = os.path.join(img_dir, 'subset_train')
        val_dir = os.path.join(img_dir, 'subset_val')
        test_dir = os.path.join(img_dir, 'subset_test')
        label_df = pd.read_csv(os.path.join(img_dir, 'subset_labels.csv'))
    else:
        train_dir = os.path.join(img_dir, 'train')
        val_dir = os.path.join(img_dir, 'val')
        test_dir = os.path.join(img_dir, 'test')
        label_df = pd.read_csv(os.path.join(img_dir, 'labels.csv'))

    ## calculate means and standard deviations per channel
    means = calc_mean(train_dir, input_height, input_width, clipping_values, channels)
    stds = calc_std(means, train_dir, input_height, input_width, clipping_values, channels)
    print(means, stds)
    print('Calculated mean and standard deviation for each channel (for training set)')

    # set up generators for model training

    training_generator = generator(x_dir=train_dir, labels=label_df, batch_size=batch_size, means=means, stds=stds,
                                   input_height=input_height, input_width=input_width, clipping_values=clipping_values,
                                   channels=channels, channel_size=channel_size)
    # Check if shape is correct
    print('Created x and y for training data')
    for data_batch, labels_batch in training_generator:
        print('This is the shape of the training data batch:', data_batch.shape)
        print('This is the shape of the training label batch:', labels_batch.shape)
        samples = data_batch
        break

    validation_generator = generator(x_dir=val_dir, labels=label_df, batch_size=batch_size, means=means, stds=stds,
                                     input_height=input_height, input_width=input_width,
                                     clipping_values=clipping_values,
                                     channels=channels, channel_size=channel_size)

    print('The minimum value is', np.min(samples))
    print('The maximum value is', np.max(samples))
    print('The mean is', np.mean(samples))
    print('The standard deviation is', np.std(samples))

    # plt.hist(samples.flatten(), bins=100)
    # plt.show()

    # generate datasets
    train_generator_func = partial(generator, train_dir, label_df, batch_size, means, stds, input_height, input_width,
                                   clipping_values, channels, channel_size, 1)

    train_ds = Dataset.from_generator(generator=train_generator_func,
                                      output_types=(tf.float32, tf.float32),
                                      output_shapes=((batch_size, input_width, input_height, channel_size),
                                                     (batch_size,)),
                                      )

    # This part generates the validation generator for the NN
    val_generator_func = partial(generator, val_dir, label_df, batch_size, means, stds, input_height, input_width,
                                 clipping_values, channels, channel_size, 1)

    val_ds = Dataset.from_generator(generator=val_generator_func,
                                    output_types=(tf.float32, tf.float32),
                                    output_shapes=((batch_size, input_width, input_height, channel_size),
                                                   (batch_size,)),
                                    )

    # Load VGG19 model
    hyperspectral_model_obj = VVG19_hyperspectral(img_w=input_width,
                                                  img_h=input_height,
                                                  channels=channel_size
                                                  )
    vgg19_template = hyperspectral_model_obj.load_vgg19_hyperspectral_template()

    # adjust to hyperspectral input
    model = hyperspectral_model_obj.resize_weights(vgg19_template)

    model.compile(optimizer=optimizers.Adam(learning_rate=2e-4),
                  loss='mean_squared_error')
    # Fit and train model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100)
    print(history.history)

    # This part generates the test generator for the NN
    test_generator_func = partial(generator, test_dir, label_df, batch_size, means, stds, input_height, input_width,
                                  clipping_values, channels, channel_size, 1)
    test_ds = Dataset.from_generator(generator=test_generator_func,
                                     output_types=(tf.float32, tf.float32),
                                     output_shapes=((batch_size, input_width, input_height, channel_size),
                                                    (batch_size,)),
                                     )
    # Evaluate on testset
    MSE = model.evaluate(test_ds)

    # Return Mean Squared Error
    print('Model Loss is {}'.format(MSE))


if __name__ == '__main__':
    main(img_dir='/mnt/datadisk/data/Sentinel2/preprocessed/asset/urban_rural',
         subset=True,
         input_height=2003,
         input_width=2003,
         channels=[],
         channel_size=13,
         clipping_values=[0, 3000],
         batch_size=20)
