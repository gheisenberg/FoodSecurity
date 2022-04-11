import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from data_utils import generator, create_splits
import matplotlib.pyplot as plt
from functools import partial
from operator import itemgetter
import pandas as pd
import numpy as np

import keras
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras import optimizers, models
from sklearn.model_selection import KFold

# import models
from vvg19 import VVG19_hyperspectral
from resnet50 import ResNet50_hyperspectral

import wandb
from wandb.keras import WandbCallback


def main(img_dir: str, csv_path: str, model_name: str, k: int, input_height: int, input_width: int, img_source: str,
         urban_rural: str, channel_size: int, batch_size: int, epochs: int, subset: bool):
    # subset: bool, clipping_values: list, channels: list,
    '''

    Args:
        img_dir (str): Path to image data
        csv_path (str): Path to cluster csv files
        model_name (str): one of ['vgg19', 'resnet50'] to choose which model is used
        k (int): number of folds for cross validation
        subset (bool): Whether or not to use a subset to test the process
        input_height (int): pixel height of input
        input_width (int): pixel width of input
        img_source (str): one of ['s2', 'viirs'] to choose whether sentinel2 or viirs (nightlight) data is used
        urban_rural (str): on of ['u','r','ur'] to choose whether only urban/rural clusters or all data is used
        channels (list):  Channels to use; [] to use all channels
        channel_size (int): Number of channels (3 for RGB, 13 for all channels)
                            !Nightlight channel is transformed to 3 channels for model compatibility
        batch_size (int): Size of training batches
        epochs (int): Number of Training Epochs
        subset (bool): Whether or not to use a subset (for testing)

    Returns:

    '''
    X_train_val, X_test, y_train_val, y_test = create_splits(img_dir, csv_path, urban_rural, subset)
    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    for fold, (train_index, val_index) in enumerate(kf.split(X_train_val)):
        print(f'Fold: {fold}')
        wandb.init(project="Asset_Wealth", entity="piastoermer", dir='/mnt/datadisk/data/Sentinel2/',
                   group=f'{model_name}_pretrained_model_{urban_rural}_{img_source}', job_type='train',
                   name=f'{model_name}_pretrained_model_{urban_rural}_{img_source}_fold_{fold}')
        config = wandb.config  # Config is a variable that holds and saves hyperparameters and inputs
        config.learning_rate = 2e-4
        config.batch_size = batch_size
        config.epochs = epochs
        config.img_width = input_width
        config.img_height = input_height
        config.model_name = 'vgg19'
        config.pretrain_weights = 'imagenet'
        config.urban_rural = urban_rural  # 'all'
        config.image_source = img_dir
        config.loss = 'mean_squared_error'
        config.metrics = ['mean_squared_error',
                          'mean_absolute_error',
                          'mean_absolute_percentage_error']

        # Load VGG19 model

        X_train, X_val = list(itemgetter(*train_index)(X_train_val)), list(itemgetter(*val_index)(X_train_val))
        y_train, y_val = y_train_val[train_index], y_train_val[val_index]
        print(f'Training size: {len(X_train)} \n Validation size: {len(X_val)} \n Test Size: {len(X_test)}')
        # generate datasets
        train_generator_func = partial(generator, img_dir, X_train, y_train, batch_size, input_height, input_width,
                                       channel_size)

        train_ds = Dataset.from_generator(generator=train_generator_func,
                                          output_types=(tf.float64, tf.float64),
                                          output_shapes=((batch_size, input_width, input_height, channel_size),
                                                         (batch_size,)),
                                          )

        # This part generates the validation generator for the NN
        val_generator_func = partial(generator, img_dir, X_val, y_val, batch_size, input_height, input_width,
                                     channel_size)

        val_ds = Dataset.from_generator(generator=val_generator_func,
                                        output_types=(tf.float64, tf.float32),
                                        output_shapes=((batch_size, input_width, input_height, channel_size),
                                                       (batch_size,)),
                                        )

        # adjust to hyperspectral input
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print("Num GPUs Available: ", len(gpus))
        mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1",
                                                                    "/gpu:2"])
        with mirrored_strategy.scope():
            if model_name == 'vgg19':
                hyperspectral_model_obj = VVG19_hyperspectral(img_w=input_width,
                                                              img_h=input_height,
                                                              channels=channel_size
                                                              )
                model = hyperspectral_model_obj.load_vgg19()
            elif model_name == 'resnet50':
                hyperspectral_model_obj = ResNet50_hyperspectral(img_w=input_width,
                                                                 img_h=input_height,
                                                                 channels=channel_size
                                                                 )
                model = hyperspectral_model_obj.load_resnet50()
            model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0001),
                          loss='mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError(),
                                                              tf.keras.metrics.MeanAbsoluteError(),
                                                              tf.keras.metrics.MeanAbsolutePercentageError(),
                                                              tf.keras.metrics.RootMeanSquaredError(),
                                                              tf.keras.metrics.CosineSimilarity()
                                                              ])
        print('Start Model Training')
        # Fit and train model
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=1,
            callbacks=[WandbCallback()])
        print(history.history)
        # Save best instance of the model.
        model.save(f'./models/{model_name}/pretrained_model_{urban_rural}_{img_source}_fold_{fold}.h5')

        # Evaluate model on hold out validation set for this fold.

        print('Evaluating Model')

        # This part generates the test generator for the NN
        test_generator_func = partial(generator, img_dir, X_test, y_test, batch_size, input_height, input_width,
                                      channel_size)
        test_ds = Dataset.from_generator(generator=test_generator_func,
                                         output_types=(tf.float64, tf.float32),
                                         output_shapes=((batch_size, input_width, input_height, channel_size),
                                                        (batch_size,)),
                                         )
        # Evaluate on testset
        evaluation = model.evaluate(test_ds)

        wandb.log({'MeanSquaredError': evaluation[0]})
        wandb.log({'MeanAbsoluteError': evaluation[1]})
        wandb.log({'MeanAbsolutePercentageError': evaluation[2]})
        wandb.log({'RootMeanSquaredError': evaluation[3]})
        wandb.log({'CosineSimilarity': evaluation[4]})

        del model
        del train_ds
        del val_ds
        del test_ds

    wandb.finish()


if __name__ == '__main__':
    main(img_dir='/mnt/datadisk/data/Sentinel2/preprocessed/asset/urban/',
         csv_path='/home/stoermer/Sentinel/gps_csv/',
         model_name='vgg19',
         k=5,
         input_height=400,
         input_width=400,
         img_source='s2',
         urban_rural='u',
         channel_size=13,
         batch_size=16,
         epochs=20,
         subset=False)
