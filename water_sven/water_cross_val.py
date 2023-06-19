#!/usr/bin/env python
# coding: utf-8

###general imports
import os

# turn off cryptic warnings, Note that you might miss important warnings! If unexpected stuff is happening turn it on!
# https://github.com/tensorflow/tensorflow/issues/27023
# Thanks @Mrs Przibylla
# '1' = Infos, '2' = warnings, '3' = Errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import shutil
import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers, layers
from tensorflow.keras import callbacks
from tensorflow.keras.layers.experimental import preprocessing
import pickle
import time
from functools import partial
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.stats as stats
import tensorflow_addons as tfa
import scipy
import math
from math import radians, cos, sin, asin, sqrt

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()
# warnings.filterwarnings("ignore")
# tf.autograph.set_verbosity(1)

###priv imports
import config as cfg
import visualizations
import nn_models as nnm
import helper_utils as hu
import geo_utils as gu

logger = hu.setup_logger(cfg.logging)
logger.warning('Logging level: ' + cfg.logging)

#####This needs to be done before using tensorflow or keras (e.g. importing losses in config)
###Some general information and settings
# print some general information
logger.info('tf keras v %s', tf.keras.__version__)
logger.info('tf v %s', tf.__version__)
# to do: try non eager execution graph?
logger.info('tf eager execution %s', tf.executing_eagerly())
# force channels-first ordering
K.set_image_data_format('channels_first')
logger.info('Image data format %s', K.image_data_format())

# do not assign complete gpu-memory but grow it as needed
# allows to run multiple models at once (otherwise whole gpu memory gets allocated/gpu gets blocked)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        raise e

###Define Model and MirroredStrategy (for mulit-gpu usage) - shall be initiated at the beginning of the program
# Create a MirroredStrategy and pass the GPUs
gpus = ["GPU:" + str(i) for i in cfg.gpus]
# https://keras.io/guides/distributed_training/
logger.debug('gpus %s', gpus)
strategy = tf.distribute.MirroredStrategy(gpus)
logger.info('Number of GPUs MirroredStrategy: {}'.format(strategy.num_replicas_in_sync))


def create_run_folders(augmentation_d, img_path, prj_path, trainHistory_subname, label_name=False, split=False,
                       height=False,
                       width=False, test_mode=False):
    """Creates a name for the model run by importing settings from config.py.
    This will also be used as the folder name where all files of this run are saved.
    Creates this folder and further subfolders.

    Note that config.py also gets replicated into the run_path for easy replication and exact configuration.
    Also imports some parameters directly from config.py

    Args:
        augmentation_d (dict): Augmentation parameters (as defined in config.py)
        normalization_d (dict): Normalization parameters (as defined in config.py)
        prj_path (str): Path where the project is located
        trainHistory_subname (str): Name of the folder where all model runs will are saved

    Returns:
        train_history_path (str): Path where all model runs are saved
        run_path (str): Path where all model run files are saved (subfolder of train_history_path)
        modelcheckpoint_path (str): Path where all modelcheckpoint files are saved (subfolder of run_path)
        augmented_images_path (str): Path where augmented images get saved in jpg-format for easy interpretation of
            input and augmentation (if activated in config.py) (subfolder of run_path)
    """

    ###Create global paths
    # Create train history path name //Run name
    img_p = os.path.basename(img_path[:-1])
    img_p = img_p.replace(',', '_')
    img_p = img_p.replace(' ', '_')
    aug_name = '_'
    for k, v in augmentation_d.items():
        if v is False or v == 0 or v == {}:
            pass
        elif v is True:
            aug_name += k + '_'
        else:
            aug_name += k + str(v)
    # replace some special chars and omit some strings for better readability and acceptable length
    aug_name = aug_name.replace(',', '_')
    aug_name = aug_name.replace('range', '')
    aug_name = aug_name.replace('_center_featurewise_std_normalization_', '')
    aug_name = aug_name.replace('_center_samplewise_std_normalization_', '')
    # strip every char containing harmful chars for paths
    aug_name = aug_name.translate({ord(c): None for c in '[],;!@#$() '})

    # create base name
    run_name = img_p + cfg.model_name + '_w' + \
               str(cfg.cnn_settings_d['weights']) + '_unfl' + str(cfg.unfreeze_layers_perc) + \
               '_d' + str(cfg.dropout_top_layers) + '_lr' + str(cfg.lr)
    if cfg.auto_adjust_lr[0]:
        run_name += '_adjustlr'
    if cfg.momentum:
        run_name += '_momentum' + str(cfg.momentum)
    run_name += '_optimizer' + cfg.optimizer + aug_name
    run_name += '_m_' + cfg.model_name
    run_name += '_h' + str(height) + '_w' + str(width)

    sample_string = label_name
    char_to_replace = {' ': '_',
                       '<': '_',
                       '=': '_',
                       ':': '_',
                       '(': '',
                       ')': ''}
    # Iterate over all key-value pairs in dictionary
    for key, value in char_to_replace.items():
        # Replace key character with value character in string
        sample_string = sample_string.replace(key, value)
        run_name = run_name.replace(key, value)
        split = split.replace(key, value)

    if test_mode:
        trainHistory_subname += 'test_mode/'
    # Create paths II
    paths_dict = {'trainH_path': ['base_path', trainHistory_subname, False],
                  'train_history_path': ['trainH_path', split, False],
                  'run_path': ['train_history_path', run_name, True],
                  'modelcheckpoint_path': ['run_path', 'modelcheckp/', False],
                  'augmented_imgs_path': ['run_path', 'augmented_imgs/', False]}

    [trainH_base_p, train_history_path, run_path, modelcheckpoint_path, augmented_images_path] = \
        hu.paths_from_base_path(prj_path, paths_dict, add_d={'train_history_path': sample_string + '/'})
    run_name += run_path[-2:]
    return train_history_path, run_path, modelcheckpoint_path, augmented_images_path, run_name


def optimizer_loading():
    """Summarizing optimizer loading to simplify code in main

    Loads optimizer (str) from config.py and

    Returns:
        optimizer (Keras optimizer): ...
    """
    # Optimizers
    if cfg.optimizer == "SGD":
        # from transferLearning.py
        optimizer = tf.keras.optimizers.SGD(learning_rate=cfg.lr, momentum=cfg.momentum, nesterov=False,
                                            name="SGD")
    elif cfg.optimizer == "Adam":
        # from transferLearning.py
        optimizer = optimizers.Adam(lr=cfg.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    else:
        raise NotImplementedError('cfg.optimizer is', cfg.optimizer,
                                  'it is not implemented right now though')
    return optimizer


def callback_loader(run_path, modelcheckpoint_path, metric):
    """Summarizing callback loading to simplify code in main

    Loads auto_adjust_lr, early_stopping and tensorboard variables from config.py

    Args:
        run_path (str): Path where run files are stored
        modelcheckpoint_path (str) Path where model checkpoints are stored

    Returns:
        callbacks_l (list): List of callback objects as passed to model.fit()
    """
    # Callbacks
    callbacks_l = []
    if cfg.auto_adjust_lr[0]:
        # Reduces the learning rate by some factors if the validation loss stops improving in some epochs
        reduce = callbacks.ReduceLROnPlateau(
            monitor='val_' + metric.name, factor=cfg.auto_adjust_lr[2],
            patience=cfg.auto_adjust_lr[1], mode='auto')
        callbacks_l.append(reduce)
    if cfg.early_stopping[0]:
        # Stops the training process if the validation loss stops improving in some epochs (here 10)
        early = callbacks.EarlyStopping(monitor='val_' + metric.name,
                                        min_delta=1e-4, patience=cfg.early_stopping[1],
                                        mode='auto')  # mode=max, restore_best_weights=True
        callbacks_l.append(early)
    if cfg.tensorboard:
        callbacks_l.append(callbacks.TensorBoard(log_dir=run_path,
                                                 histogram_freq=0,
                                                 write_graph=True,
                                                 write_images=False,
                                                 write_steps_per_second=True,
                                                 update_freq="epoch",
                                                 profile_batch=0,
                                                 embeddings_freq=0,
                                                 embeddings_metadata=None, ))

    callbacks_l.append(callbacks.ModelCheckpoint(modelcheckpoint_path,
                                                 monitor='val_' + metric.name,
                                                 verbose=1,
                                                 save_weights_only=True,
                                                 save_best_only=True))
    return callbacks_l


def model_loader(num_labels, type_m, shape, **aug_d):
    """Summarizing model loading from string defined in config.py to simplify code in main
    Also loads lots of variables from config.py, especially: cnn_settings_d (and many more)
    If defined in config.py add_custom_top_layers it will also add custom top layers by calling add_top_layer from
    nn_models.py!

    Returns:
        model (Keras model): e.g. VGG19 or resnet50 if defined with custom top layers
    """
    cnn_settings_d = cfg.cnn_settings_d
    cnn_settings_d['input_shape'] = shape
    logger.debug('cnn settings d', cnn_settings_d)
    if cfg.model_name == 'resnet152':
        base_model = tf.keras.applications.ResNet152V2(**cnn_settings_d)
    elif cfg.model_name == 'resnet50':
        base_model = tf.keras.applications.ResNet50V2(**cnn_settings_d)
    elif cfg.model_name == 'vgg16':
        base_model = tf.keras.applications.VGG16(**cnn_settings_d)
    elif cfg.model_name == 'vgg19':
        base_model = tf.keras.applications.VGG19(**cnn_settings_d)
    elif cfg.model_name == 'inceptionv3':
        base_model = tf.keras.applications.InceptionV3(**cnn_settings_d)
    elif cfg.model_name == 'densenet121':
        base_model = tf.keras.applications.DenseNet121(**cnn_settings_d)
    elif cfg.model_name == 'densenet201':
        base_model = tf.keras.applications.DenseNet201(**cnn_settings_d)
    elif cfg.model_name == 'xception':
        base_model = tf.keras.applications.Xception(**cnn_settings_d)
    else:
        raise NotImplementedError("The model you want is not implemented right now", cfg.model_name)

        # base_model = Model(inputs, outputs)
    # Add custom Top Layers (Imagenet has 1000nds of classes - we need 3!)
    if cnn_settings_d['include_top'] is False and cfg.add_custom_top_layers:
        model = nnm.add_classification_top_layer(base_model, num_labels, cfg.neurons_l, type_m,
                                                 cfg.dropout_top_layers, cfg.unfreeze_layers_perc)
        logger.info(f'Final Model with added top layers {type(base_model)} layers {len(base_model.layers)}'
                    f'of which {round(len(base_model.layers) * cfg.unfreeze_layers_perc / 100)} are unfroozen or'
                    f'{cfg.unfreeze_layers_perc} %')
    else:
        model = base_model
    # Add preprocessing layers
    if aug_d:
        input_layer = tf.keras.Input(shape=shape)
        aug_layer = augmentation(**aug_d)
        x = aug_layer(input_layer)
        x = model(x)
        model = tf.keras.Model(inputs=input_layer, outputs=x)
        logger.debug('final model with aug', model)
    return model


def dataset_creator(train_df, validation_df, test_df, prediction_type, num_labels, cache_p, input_shape, channel_l,
                    **augargs):
    ds_l = []
    for typ, df in zip(['train', 'validation', 'test'], [train_df, validation_df, test_df]):
        if typ == 'train':
            steps_per_epoch_train = math.ceil(len(df) / cfg.batch_size)
        elif typ == 'validation':
            steps_per_epoch_val = math.ceil(len(df) / cfg.batch_size)
        elif typ == 'test':
            steps_per_epoch_test = math.ceil(len(df) / cfg.batch_size)
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
        if cfg.verbose:
            logger.debug('files ds %s', files)
            logger.debug('labels ds %s', labels)
        func = partial(gu.load_geotiff, height=input_shape[1], width=input_shape[2], channel_l=channel_l,
                       only_return_array=True)
        output_signature = tf.TensorSpec(shape=input_shape, dtype=tf.float32)
        ds = files.map(lambda x: tf.py_function(func, [x], output_signature),
                       num_parallel_calls=tf.data.AUTOTUNE)
        if cfg.verbose:
            logger.debug('Read imgs ds %s', ds)
        assert len(ds) == len(labels)
        ds = tf.data.Dataset.zip((ds, labels))
        if cfg.verbose:
            logger.debug('Zipped ds %s %s', ds, len(ds))
        # significantly improves speed! ~25%
        if typ == 'train':
            cache_f = cache_p + typ
            ds = ds.cache(cache_f)
            # buffer_size does not seem to influence the performance
            # to do: set on True again?!
            ds = ds.shuffle(cfg.batch_size, reshuffle_each_iteration=True)
        ds = ds.batch(cfg.batch_size)
        # does not seem to have an influence when used with cached datasets
        ds = ds.prefetch(tf.data.AUTOTUNE)
        ds_l.append(ds)
    if cfg.verbose:
        logger.debug('Final ds %s', ds)
        logger.debug('steps p epoch train, val, test %s %s %s',
                     steps_per_epoch_train, steps_per_epoch_val, steps_per_epoch_test)
    return ds_l[0], ds_l[1], ds_l[2], steps_per_epoch_train, steps_per_epoch_val, steps_per_epoch_test


def special_replace(row, replace_d, col, labels_df, drop):
    """Experimental and seldom used: replacing labels by others in"""
    replaced = False
    if 'urban' in cfg.labels_f:
        if col == 'source of drinking water (weights): max':
            if row[0] == 1.0:
                replace_d[row[0]] = 2.0
                replaced = True
                logger.info('replaced label:', row[0], 'n:', row[1][col], '< threshold', len(labels_df) * drop,
                            'with 2')
    return replace_d, replaced


def load_labels(file, col, run_path, base_path=False, mode='categorical', drop=0.05, min_year=False,
                split='split: random all', map_drop_to_other=False, special_rep=False, break_v=2000,
                **kwargs):
    """Loads and processes labels from a CSV file.

    Args:
        file (str): Path to the CSV file containing the labels: Following columns are required: ['GEID', 'DHSID',
        split, col, "households", "adm0_name", "adm1_name", "adm2_name", "LATNUM", "LONGNUM", "DHSYEAR"]
        col (str): Name of the column containing the labels.
        run_path (str): Path for saving visualizations.
        base_path (str, optional): Base path for checking if there are images, if not provided defaults to False.
        mode (str, optional): Mode of processing, can be 'categorical' or 'regression'. Defaults to 'categorical'.
        drop (float, optional): Threshold for dropping labels in 'categorical' mode based on their frequency. Defaults to 0.05.
        min_year (int, optional): Minimum year for data filtering, if not provided defaults to False.
        split (str, optional): The column in the file defining data splits. Defaults to 'split: random all'.
        map_drop_to_other (bool, optional): In 'categorical' mode: If True, maps the dropped labels to 'other'. Defaults to False.
        special_rep (bool, optional): If True, applies a special replacement for dropped labels. Defaults to False.
        break_v (int, optional): Threshold for warning about the number of labels. Defaults to 2000.
        **kwargs: Additional keyword arguments:
        'min std': Float: maps (or drops) labels below the value to min value
        'max std': Float: maps (or drops) labels above the value to max value
        'drop min max value': use 'True' to drop the values below/above the 'min value' and 'max value'
        'reverse label': use 'True' to multiply label by -1
        'label normalization': 'Z' or '0,1' label normalization
        'label transform': 'boxcox' transforms the label by boxcox transformation (note 0 is shifted to 0.1)

    Returns:
        list: DataFrame objects for each split in the data (train/validation or splits).
        DataFrame: DataFrame containing the test split
        DataFrame: DataFrame containing the full data
        dict: Mapping of original to new labels if mode is 'categorical'.
        dict: Dictionary of additional parameters computed during processing.
        object: Scaler object used for label normalization in 'regression' mode, False otherwise.
    """

    reverse_label, min_std, max_std, drop_min_max_value, normalization, transform = False, False, False, False, \
        False, False
    for k, v in kwargs.items():
        if k == 'reverse label':
            reverse_label = v
        elif k == 'min std':
            min_std = v
        elif k == 'max std':
            max_std = v
        elif k == 'drop min max value':
            drop_min_max_value = v
        elif k == 'label normalization':
            normalization = v
        elif k == 'label transform':
            transform = v
    if drop_min_max_value:
        logger.info('dropping labels %s', drop_min_max_value)
    if min_std or max_std:
        logger.info('min/max value %s %s', min_std, max_std)
    labels_df = pd.read_csv(file)
    mixed_columns = [col for col in labels_df.columns if labels_df[col].apply(type).nunique() > 1]
    logger.debug('Mixed columns %s', mixed_columns)
    # shuffle in pandas ##doesnt effect splits
    labels_df = labels_df.sample(frac=1)

    # delete everything where no split is defined
    # note: in many implementations, the split will make a check for urban/rural, GTIF images and a min_year obsolete
    # to be sure it is still implemented here, though
    if split:
        labels_df = labels_df.dropna(subset=[split])
    if base_path:
        # create pathes
        labels_df['path'] = base_path + labels_df['GEID'] + labels_df['DHSID'].str[-8:] + '.tif'
        available_files = hu.files_in_folder(base_path)
        # check if actually available
        labels_df["path"] = labels_df['path'].apply(lambda x: x if x in available_files else np.NaN)
    if min_year:
        logger.info(
            f"If not using min year {min_year} {len(labels_df[labels_df['DHSYEAR'] < min_year])} more data could be "
            f"used")
        labels_df = labels_df[labels_df['DHSYEAR'] >= min_year]
    if cfg.test_mode:
        labels_df = labels_df[:cfg.test_mode]
    # create visualization for incoming labels before dropping values
    visualizations.standard_hist_from_df(labels_df[col], run_path, '', title_in=col)
    visualizations.standard_hist_from_df(labels_df[col], run_path, '', title_in=col + ' wo xlim', xlim=False)

    labels_df = labels_df[['GEID', 'DHSID', 'path', split, col, "households", "adm0_name", "adm1_name", "adm2_name",
                           "LATNUM", "LONGNUM", "DHSYEAR", "URBAN_RURA"]]
    labels_df['label'] = labels_df[col]
    # drop all which have no label or no input
    if labels_df[['path', 'label', split]].isnull().any(axis=1).any():
        logger.warning(
            f"!!!Caution Missing values getting dropped {len(labels_df[labels_df[['path', 'label']].isnull().any(axis=1)])}"
            f" from which are {len(labels_df[labels_df[split].isna()])} not in split,"
            f"{len(labels_df[labels_df['path'].isna()])} missing files \n"
            f"{len(labels_df[labels_df['label'].isna()])} missing values (overlapping calculation!) \n"
            f"--> writing missing files into {cfg.prj_folder + 'missing_files.csv'} "
            f" if these have labels")

        if not labels_df[(labels_df['path'].isna()) & (labels_df['label'].notna())].empty:
            labels_df[(labels_df['path'].isna()) & (labels_df['label'].notna())].to_csv(
                cfg.prj_folder + 'missing_files.csv', index=False)
    # drop all which have no label, no path or no split
    labels_df = labels_df.dropna(subset=['path', 'label', split])
    logger.debug('labels df after dropping na values in path, label and split\n%s', labels_df)
    logger.debug('Missing values in label df after dropping na values\n%s', labels_df[labels_df.isna().any(axis=1)])
    visualizations.standard_hist_from_df(labels_df[col], run_path, '', title_in=col + ' after dropping')

    label_mapping = {}
    add_params = {}
    scaler = False
    if cfg.verbose:
        for c in labels_df.columns:
            logger.debug('labels_df value counts %s', labels_df[c].value_counts(dropna=False))
    if mode == 'categorical':
        replace_d = {}
        values_df = pd.DataFrame(labels_df[col].value_counts())
        for nr, row in enumerate(values_df.iterrows()):
            if row[1][col] > len(labels_df) * drop:
                replace_d[row[0]] = nr
                if cfg.verbose:
                    logger.debug('label:', row[0], 'n:', row[1][col])
            else:
                replaced = False
                if special_rep:
                    replace_d, replaced = special_replace(row, replace_d, col, labels_df, drop)
                if map_drop_to_other and not replaced:
                    replace_d[row[0]] = 'other'
                    if cfg.verbose:
                        logger.debug('replaced label:', row[0], 'n:', row[1][col], '< threshold', len(labels_df) * drop,
                                     'with "other"')
                else:
                    replace_d[row[0]] = np.NaN
                    if cfg.verbose:
                        logger.debug('dropped label:', row[0], 'n:', row[1][col], '< threshold', len(labels_df) * drop)
        labels_df = labels_df.replace({'label': replace_d})
        labels_df = labels_df.dropna(subset=['label'])
        labels_df['label'] = labels_df['label'].astype(int)
        label_mapping = {v: k for k, v in replace_d.items() if not np.isnan(v)}
        # drop
    elif mode == 'regression':
        logger.debug('Some basic statistics before normalization/transformation:')
        add_params = {'mean: DF': labels_df['label'].mean(), 'std: DF': labels_df['label'].std(),
                      'skew: DF': labels_df['label'].skew(), 'size: DF': len(labels_df),
                      'kurtosis: DF': labels_df['label'].kurtosis()}
        logger.debug('min %s', min(labels_df['label']))
        logger.debug('max %s', max(labels_df['label']))
        logger.debug('std deviation %s', labels_df['label'].std())
        logger.debug('mean %s', labels_df['label'].mean())
        logger.debug('skew %s', labels_df['label'].skew())

        if reverse_label:
            labels_df['label'] = labels_df['label'] * (-1.0)
            labels_df['reversed'] = labels_df['label']
            # min max value (dropping or setting to min/max value)

        if min_std or max_std:
            logger.info('min/max value: %s %s %s %s %s %s', min_std, max_std, 'dropping min/max:',
                        labels_df['label'].min(),
                        labels_df['label'].max())
            visualizations.standard_hist_from_df(labels_df['label'], run_path, '',
                                                 title_in='Before removing MIN MAX', xlim=False)
            mean = labels_df['label'].mean()
            stda = labels_df['label'].std()
            if drop_min_max_value:
                if max_std:
                    labels_df = labels_df[labels_df['label'] <= mean + max_std * stda]
                if min_std:
                    labels_df = labels_df[labels_df['label'] >= mean - min_std * stda]
            else:
                if max_std:
                    labels_df.loc[labels_df['label'] >= mean + max_std * stda, 'label'] = \
                        mean + max_std * stda
                if min_std:
                    labels_df.loc[labels_df['label'] <= mean - min_std * stda, 'label'] = \
                        mean - min_std * stda

        if transform == 'boxcox':
            # boxcox doesnt work with 0
            # replace_d = {'label': {0: 0.1}}
            min_v = min(labels_df['label'])
            if min_v <= 0:
                shift_v = abs(min_v) + 1
                add_params['shift value'] = shift_v
                # right shift values (boxcox cant handle negative or 0 values)
                labels_df['label'] = labels_df['label'] + shift_v
            # labels_df = labels_df.replace(replace_d)
            array, lmbda = stats.boxcox(labels_df['label'])
            add_params['lmbda'] = lmbda
            labels_df['label'] = array
        elif transform:
            raise NotImplementedError('Transform not implemented:', transform)

        if transform:
            add_params['skew: after transforming'] = labels_df['label'].skew()
            add_params['mean: after transforming'] = labels_df['label'].mean()
            add_params['std: after transforming'] = labels_df['label'].std()
            add_params['kurtosis: after transforming'] = labels_df['label'].kurtosis()
            logger.info('Some basic statistics after transforming w', transform, ':')
            logger.info('min %s', min(labels_df['label']))
            logger.info('max %s', max(labels_df['label']))
            logger.info('std deviation %s', labels_df['label'].std())
            logger.info('mean %s', labels_df['label'].mean())
            logger.info('skew %s', labels_df['label'].skew())
            labels_df['transformed'] = labels_df['label']
            visualizations.standard_hist_from_df(labels_df['transformed'], run_path, '', title_in='Transformation')

        if normalization:
            if normalization == '0,1':
                scaler = MinMaxScaler()
                labels_df['label'] = scaler.fit_transform(np.array(labels_df['label']).reshape(-1, 1))
            elif normalization == 'Z':
                scaler = StandardScaler()
                labels_df['label'] = scaler.fit_transform(np.array(labels_df['label']).reshape(-1, 1))
            else:
                raise NotImplementedError
            add_params['skew: df (normalized)'] = labels_df['label'].skew()
            add_params['mean: df (normalized)'] = labels_df['label'].mean()
            add_params['std: df (normalized)'] = labels_df['label'].std()
            add_params['kurtosis: df (normalized)'] = labels_df['label'].kurtosis()
            logger.info('Some basic statistics after %s %s', normalization, 'normalization:')
            logger.info('min %s', min(labels_df['label']))
            logger.info('max %s', max(labels_df['label']))
            logger.info('std deviation %s', labels_df['label'].std())
            logger.info('mean %s', labels_df['label'].mean())
            logger.info('skew %s', labels_df['label'].skew())
            labels_df['normalized'] = labels_df['label']
            visualizations.standard_hist_from_df(labels_df['normalized'], run_path, '', title_in='Normalization')
        else:
            logger.info('Did not normalize')

    # train_df = labels_df[labels_df[split] == 'train']
    # validation_df = labels_df[labels_df[split] == 'validation']
    split_dfs = []
    test_df = False
    for spl in labels_df[split].unique():
        logger.debug(spl)
        spl_df = labels_df[labels_df[split] == spl]
        add_params['mean: ' + spl] = spl_df['label'].mean()
        add_params['std: ' + spl] = spl_df['label'].std()
        add_params['skew: ' + spl] = spl_df['label'].skew()
        add_params['size: ' + spl] = len(spl_df)
        add_params['kurtosis: ' + spl] = spl_df['label'].kurtosis()

        if spl == 'test':
            test_df = spl_df
            if abs(test_df['label'].mean() - labels_df['label'].mean()) > 0.05:
                logger.warning('test df mean is too different from whole df mean')
                logger.warning('test df mean: %s', test_df['label'].mean())
                logger.warning('whole df mean: %s', labels_df['label'].mean())
                # input('Are you sure you want to continue?')
            if abs(test_df['label'].std() - labels_df['label'].std()) > 0.1:
                logger.warning('test df std is too different from whole df std')
                logger.warning('test df std: %s', test_df['label'].std())
                logger.warning('whole df std: %s', labels_df['label'].std())
                # input('Are you sure you want to continue?')
            if abs(test_df['label'].skew() - labels_df['label'].skew()) > 0.1:
                logger.warning('test df skew is too different from whole df skew')
                logger.warning('test df skew: %s', test_df['label'].skew())
                logger.warning('whole df skew: %s', labels_df['label'].skew())
                # input('Are you sure you want to continue?')
            if abs(test_df['label'].kurtosis() - labels_df['label'].kurtosis()) > 0.1:
                logger.warning('test df kurtosis is too different from whole df kurtosis')
                logger.warning('test df kurtosis: %s', test_df['label'].kurtosis())
                logger.warning('whole df kurtosis: %s', labels_df['label'].kurtosis())
                # input('Are you sure you want to continue?')
            # if cfg.test_mode:
            #     test_df = spl_df[:math.floor(cfg.test_mode / 10)]
        else:
            # if cfg.test_mode:
            #     spl_df = spl_df[:math.floor(cfg.test_mode / 5)]
            #     logger.warning('testing mode!!! Limited train df: %s', len(spl_df) * 5)
            split_dfs.append(spl_df)
        visualizations.standard_hist_from_df(spl_df['label'], run_path, 'Label', title_in=spl)
    visualizations.standard_hist_from_df(labels_df['label'], run_path, '', title_in='Label')
    visualizations.standard_hist_from_df(labels_df[labels_df['DHSYEAR'] < 2015]['label'], run_path, '',
                                         title_in='Label before 2015')
    visualizations.standard_hist_from_df(labels_df[labels_df['DHSYEAR'] >= 2015]['label'], run_path, '',
                                         title_in='Label since 2015')

    logger.debug('Input Df')
    logger.debug(labels_df['label'])
    logger.debug(labels_df.count())
    spl_l = [[s[split].iloc[0] for s in split_dfs], split_dfs]
    if test_df:
        spl_l[0].append(test_df[split].iloc[0])
        spl_l[1].append(test_df)
    for st, df in zip(spl_l[0], spl_l[1]):
        logger.debug(st)
        logger.debug(df)
        if len(df) / len(labels_df) <= 0.08 or len(df) / len(labels_df) >= 0.22:
            logger.warning('%s df\n %s', st, df)
            logger.warning(f'CAUTION!!!: DF has critically low or high amount of values {len(df) / len(labels_df)} \n')
    if len(labels_df['label']) <= break_v:
        logger.warning('WARNING!!!: length of labels is below %s %s %s', break_v, 'len labels', len(labels_df['label']))
        input('Are you sure you want to continue?')
    # sort add params
    add_params = {k: v for k, v in sorted(add_params.items(), key=lambda item: item[0])}
    return split_dfs, test_df, labels_df, label_mapping, add_params, scaler


def reverse_norm_transform(df, normalization, transform, scaler, additional_params, run_path, **kwargs):
    for k, v in kwargs.items():
        if k == 'label normalization':
            normalization = v
        elif k == 'label transform':
            transform = v
    logger.info('transform', transform)
    logger.info('normalization', normalization)
    new_df = df.copy()
    for col in new_df.columns:
        if normalization:
            if normalization == 'Z':
                new_df[col] = scaler.inverse_transform(np.array(new_df[col]).reshape(-1, 1))
                # reverse_zscore(new_df[col], additional_params['normalization mean'],
                #                      additional_params['normalization standard deviation'])
            elif normalization == '0,1':
                new_df[col] = scaler.inverse_transform(np.array(new_df[col]).reshape(-1, 1))
            else:
                raise NotImplementedError('Did not implement', normalization)
            visualizations.standard_hist_from_df(new_df[col], run_path, col, title_in='Denormalized')
            # visualizations.histogram(new_df[col], 'probability', col + ' denormalized', run_path, col + ' denormalized')

        if transform:
            if transform == 'boxcox':
                new_df[col] = scipy.special.inv_boxcox(new_df[col], additional_params['lmbda'])
                if 'shift value' in additional_params:
                    new_df[col] = new_df[col] - additional_params['shift value']
            else:
                raise NotImplementedError('Did not implement', transform)
            visualizations.standard_hist_from_df(new_df[col], run_path, col, title_in='Detransformed')
            # visualizations.histogram(new_df[col], 'probability', col + ' detransformed', run_path, col + ' detransformed')
    return new_df


def augmentation(**kwargs):
    layer_l = []
    for k, v in kwargs.items():
        if k == 'horizontal_flip' and v == True:
            layer_l.append(preprocessing.RandomFlip("horizontal"))
        elif k == 'vertical_flip' and v == True:
            layer_l.append(preprocessing.RandomFlip("vertical"))
        elif k == 'zoom_range':
            layer_l.append(preprocessing.RandomZoom(height_factor=(-v, v)))
        elif k == 'rotation_range':
            layer_l.append(preprocessing.RandomRotation((-v, v)))
        else:
            raise NotImplementedError()
    data_augmentation_layers = False
    if layer_l:
        logger.info('Augmentation layers %s', layer_l)
        data_augmentation_layers = tf.keras.Sequential(layer_l)
        logger.debug('%s', data_augmentation_layers)
    return data_augmentation_layers


def evaluate_dataset(model, evaluate_ds, steps_per_epoch_evaluate, run_path, label_mapping, add_params, scaler,
                     label_d, split, evaluate_df, additional_reports_d, add_name='', evaluate_mode='test',
                     label_col_in=False):
    eval_path = os.path.join(run_path, 'splits', '')
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)
    beta, alpha, pearson_corr, rmse, nrmse = False, False, False, False, False
    logger.debug(evaluate_mode)
    logger.debug('ds %s', evaluate_ds)
    logger.debug('eval df\n%s', evaluate_df)
    logger.debug(evaluate_df.columns)
    evaluate_df = evaluate_df.copy()
    # Create Confusion matrix for the evaluate dataset
    # Get np array of predicted labels and true labels for evaluate dataset
    evaluate_pred = model.predict(evaluate_ds, steps=steps_per_epoch_evaluate)
    evaluate_true = np.concatenate([y for x, y in evaluate_ds], axis=0)
    if cfg.type_m == 'categorical':
        evaluate_pred = np.argmax(evaluate_pred, axis=1)
        evaluate_true = np.argmax(evaluate_true, axis=1)
    else:
        evaluate_pred = np.array(evaluate_pred).reshape(-1)
        evaluate_true = np.array(evaluate_true).reshape(-1)
    assert len(evaluate_pred) == len(evaluate_true)
    evaluate_df['Prediction'] = evaluate_pred
    evaluate_df['Actual'] = evaluate_df['label'].copy()
    evaluate_df[f'manual Actual true'] = evaluate_true
    logger.debug('evaluate df after prediction\n%s', evaluate_df)
    if not evaluate_df['Actual'].equals(evaluate_df[f'manual Actual true']):
        logger.debug(evaluate_df[['Actual', f'manual Actual true']])
        raise ValueError("the two different Actual values do not match...")
    evaluate_df = evaluate_df.drop("manual Actual true", axis=1)

    if cfg.type_m == 'regression':
        beta, alpha, pearson_corr, rmse, nrmse = \
            visualizations.scatterplotRegressionMultiInputs(df=evaluate_df[["Actual", "Prediction"]],
                                                            run_path=eval_path,
                                                            file_name=split + ' ' + evaluate_mode + add_name)
        for na, v in zip(["beta", "alpha", "pearson_corr", "rmse", "nrmse"],
                         [beta, alpha, pearson_corr, rmse, nrmse]):
            additional_reports_d[f"{evaluate_mode}_{na}"] = v
        if cfg.report_original_data and \
                ('label normalization' in label_d or 'transformation' in label_d):
            reversed_evaluate_prediction = reverse_norm_transform(evaluate_df[["Actual", "Prediction"]].copy(),
                                                                  cfg.label_normalization, cfg.label_transform,
                                                                  scaler,
                                                                  additional_params=add_params,
                                                                  run_path=eval_path, **label_d)
            evaluate_df["Actual Original Data"] = reversed_evaluate_prediction["Actual"]
            evaluate_df["Pred Original Data"] = reversed_evaluate_prediction["Prediction"]
            beta, alpha, pearson_corr, rmse, nrmse = \
                visualizations.scatterplotRegressionMultiInputs(
                    df=evaluate_df[["Actual Original Data", "Pred Original Data"]],
                    run_path=eval_path,
                    file_name='Original_Data_' + split + ' ' + evaluate_mode + add_name)
            for na, v in zip(["beta", "alpha", "pearson_corr", "rmse", "nrmse"],
                             [beta, alpha, pearson_corr, rmse, nrmse]):
                additional_reports_d[f"{evaluate_mode}_{na}_Original_Data"] = v
            # proven works
            # if label_col_in:
            #     beta, alpha, pearson_corr, rmse, nrmse = \
            #         visualizations.scatterplotRegression(df=evaluate_df[[label_col_in, "Actual Original Data"]],
            #                                              run_path=eval_path,
            #                                              file_name='Original_Data_compare' + split + add_name)
    evaluate_df.to_csv(eval_path + f'{evaluate_mode}_df_{split}.csv', index=False)

    sk_metrics_d = {}
    if cfg.type_m == 'categorical':
        logger.debug(len(evaluate_pred), len(evaluate_true))
        logger.debug('evaluatepred', evaluate_pred)
        logger.debug('true', evaluate_true)
        # test_prediction = np.array(test_pred)
        # test_true = np.array(test_true)
        add_params['f1 micro'] = f1_score(evaluate_true, evaluate_pred, average='micro')

        # Confusion matrix
        cm_plot_labels = list(label_mapping.values())
        visualizations.plot_CM(evaluate_true, evaluate_pred, cm_plot_labels, eval_path +
                               'ConfusionMatrix' + add_name)
        # create sklearn report (for f1 score and more)
        classification_d = classification_report(evaluate_true, evaluate_pred, output_dict=True,
                                                 target_names=cm_plot_labels)
        logger.debug('evaluate classification', classification_d)
        logger.debug('f1', add_params['f1 micro'])
        # transform for one row
        for class_name, sdic in classification_d.items():
            try:
                for metric_n, v in sdic.items():
                    sk_metrics_d[class_name + ': ' + metric_n] = v
            except AttributeError:
                sk_metrics_d[class_name] = sdic
    return additional_reports_d, sk_metrics_d, evaluate_df


def evaluate_final_dataset(test_dfs_l, val_dfs_l, run_path, split_col, labels_df=False):
    if cfg.type_m == 'regression':
        for folder in ['csv/', 'adm/', 'clustered/', 'moz/', 'clustered/scatter/', 'clustered/dbscan/',
                       'clustered/csv/']:
            if not os.path.exists(run_path + folder):
                os.makedirs(run_path + folder)

        beta, alpha, pearson_corr, rmse, nrmse = False, False, False, False, False
        eval_d = {}

        # combine val dfs and test dfs
        test_df = pd.concat(test_dfs_l)
        final_val_df = pd.concat(val_dfs_l)

        names = ['Val', 'Val: different Split Models', 'Test', 'Test: different Split Models']
        dfs = [final_val_df, final_val_df, test_df, test_df]
        combined_aggregate_l = []
        combined_aggregate_names_l = []
        for stdw in [None, 3, 2]:
            combined_df1 = test_df.copy()
            if stdw:
                combined_df1 = combined_df1[(combined_df1['Actual']
                                             <= combined_df1['Actual'].mean() + combined_df1['Actual'].std() * stdw) &
                                            (combined_df1['Actual'] >= combined_df1['Actual'].mean() - combined_df1[
                                                'Actual'].std() * stdw)]
            # names += [f'Test max STD {stdw}', f'Test different Split Models {stdw}']
            # dfs += [combined_df1, combined_df1]
            # create urban and rural and  >= 2015 < 2015 and Moz dfs
            combined_before_2015_df = combined_df1[combined_df1['DHSYEAR'] < 2015]
            combined_since_2015_df = combined_df1[combined_df1['DHSYEAR'] >= 2015]
            moz_df = combined_df1[combined_df1['adm0_name'] == 'Mozambique']
            urban_df = combined_df1[combined_df1['URBAN_RURA'] == 'U']
            moz_urban_df = urban_df[urban_df['adm0_name'] == 'Mozambique']
            rural_df = combined_df1[combined_df1['URBAN_RURA'] == 'R']
            moz_rural_df = rural_df[rural_df['adm0_name'] == 'Mozambique']
            urban_before_2015_df = urban_df[urban_df['DHSYEAR'] < 2015]
            urban_since_2015_df = urban_df[urban_df['DHSYEAR'] >= 2015]
            rural_since_2015_df = rural_df[rural_df['DHSYEAR'] >= 2015]
            rural_before_2015_df = rural_df[rural_df['DHSYEAR'] < 2015]
            names += [f'Urban STD {stdw}', f'Rural STD {stdw}', f'Mozambique STD {stdw}',
                      f'Mozambique urban STD {stdw}', f'Mozambique rural STD {stdw}', f'Before 2015 STD {stdw}',
                      f'Since 2015 STD {stdw}', f'Urban before 2015 STD {stdw}',
                      f'Urban Since 2015 STD {stdw}', f'Rural before 2015 STD {stdw}',
                      f'Rural Since 2015 STD {stdw}']
            dfs += [urban_df, rural_df, moz_df, moz_urban_df, moz_rural_df, combined_before_2015_df,
                    combined_since_2015_df, urban_before_2015_df, urban_since_2015_df, rural_before_2015_df,
                    rural_since_2015_df]

            # create aggregates
            for sub_df_ind, combined_df2 in zip(['All', 'Urban', 'Rural'], [combined_df1, urban_df, rural_df]):
                                            #make sure to get adm2/1 with matching adm0(&1)
                for adm, cluster_cols in zip(["adm2_name", "adm1_name", "adm0_name"],
                                             [["adm2_name", "adm1_name", "adm0_name"], ["adm1_name", "adm0_name"],
                                              ["adm0_name"]]):
                    for year in [False, True]:
                        add_year_str = ''
                        if year:
                            cluster_cols.append('DHSYEAR')
                            add_year_str = ' Year'
                        combined_df3 = combined_df2.copy()
                        numeric_cols = combined_df3.select_dtypes(include=[np.number]).columns.tolist()
                        for col in cluster_cols:
                            if col in numeric_cols:
                                numeric_cols.remove(col)
                        combined_aggregate_df = combined_df3.groupby(cluster_cols)[numeric_cols].mean()
                        combined_aggregate_df = combined_aggregate_df.reset_index()
                        add_df = combined_df3[cluster_cols + ['households']].copy()
                        clusters = add_df.groupby(cluster_cols).size().reset_index().rename(columns={0: 'clusters'})
                        hh = add_df.groupby(cluster_cols)['households'].sum().reset_index().rename(
                            columns={0: 'households sum'})
                        combined_aggregate_df = pd.merge(combined_aggregate_df, clusters, on=cluster_cols)
                        combined_aggregate_df = pd.merge(combined_aggregate_df, hh, on=cluster_cols)
                        combined_aggregate_df = combined_aggregate_df.reset_index()
                        combined_aggregate_l.append(combined_aggregate_df)
                        combined_aggregate_names_l.append(f"{sub_df_ind} {adm[:4]}{add_year_str} STD {stdw}")
                        if year:
                            combined_aggregate_l.append(combined_aggregate_df[combined_aggregate_df['DHSYEAR'] >= 2015])
                            combined_aggregate_names_l.append(
                                f"{sub_df_ind} {adm[:4]}{add_year_str} since 2015 STD {stdw}")
                            combined_aggregate_l.append(combined_aggregate_df[combined_aggregate_df['DHSYEAR'] < 2015])
                            combined_aggregate_names_l.append(
                                f"{sub_df_ind} {adm[:4]}{add_year_str} before 2015 STD {stdw}")
                        combined_aggregate_l.append(
                            combined_aggregate_df[combined_aggregate_df['adm0_name'] == 'Mozambique'])
                        combined_aggregate_names_l.append(
                            f"{sub_df_ind} {adm[:4]}{add_year_str} Mozambique STD {stdw}")

        # cluster stuff
        for year in ['']:
            for splitted_n, cluster_cols in zip(['only location', 'location and year'],
                                                [['clustered'], ['clustered', 'DHSYEAR']]):
                df_dict = {}
                for distance in [0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20]:
                    if not year:
                        combined_df2 = test_df.copy()
                    else:
                        combined_df2 = combined_since_2015_df.copy()
                    if sub_df_ind == 'All':
                        combined_df2.to_csv(run_path + f'clustered/csv/1a_{splitted_n}_{year}.csv')
                    clustered_df_in = gu.cluster_coordinates(combined_df2, distance)
                    if sub_df_ind == 'All':
                        clustered_df_in.to_csv(run_path + f'clustered/csv/1in_clusters_{splitted_n}_{year}.csv')
                    unclustered_df = clustered_df_in[clustered_df_in['clustered'] == -1]
                    clustered_df_in = clustered_df_in[clustered_df_in['clustered'] != -1]
                    if sub_df_ind == 'All':
                        combined_df2.to_csv(run_path + f'clustered/csv/1clustered_{splitted_n}_{year}.csv')
                    for sub_df_ind in ['All', 'U', 'R', 'U+R']:
                        #, 'R+U']:
                        indi_limit = False
                        indi_with = False
                        if 'R' == sub_df_ind[0]:
                            indi_limit = 'R'
                        if 'U' == sub_df_ind[0]:
                            indi_limit = 'U'
                        if 'U+R' == sub_df_ind or 'R+U' in sub_df_ind:
                            indi_with = True

                        # calculate different modi
                        if indi_limit and indi_with:
                            clustered_df = clustered_df_in.groupby(cluster_cols). \
                                filter(lambda x: (x['URBAN_RURA'] == indi_limit).any())
                            unclustered_df2 = clustered_df.groupby(cluster_cols).filter(lambda x: len(x) == 1)
                            clustered_df = clustered_df.groupby(cluster_cols).filter(lambda x: len(x) >= 2)
                            unclustered_df = pd.concat([unclustered_df, unclustered_df2])
                        elif indi_limit:
                            clustered_df = clustered_df_in[clustered_df_in['URBAN_RURA'] == indi_limit]
                            unclustered_df2 = clustered_df.groupby(cluster_cols).filter(lambda x: len(x) == 1)
                            clustered_df = clustered_df.groupby(cluster_cols).filter(lambda x: len(x) >= 2)
                            unclustered_df = unclustered_df[unclustered_df['URBAN_RURA'] == indi_limit]
                            unclustered_df = pd.concat([unclustered_df, unclustered_df2])
                        elif not indi_limit:
                            clustered_df = clustered_df_in.copy()
                        else:
                            raise NotImplementedError("This shouldnt happen")
                        numeric_cols = clustered_df.select_dtypes(include=[np.number]).columns.tolist()
                        for col in cluster_cols:
                            if col in numeric_cols:
                                numeric_cols.remove(col)
                        cluster_aggregate_df = clustered_df.groupby(cluster_cols)[numeric_cols].mean()
                        if sub_df_ind == 'All':
                            cluster_aggregate_df.to_csv(run_path + f'clustered/csv/1_clustered_{splitted_n}_{year}.csv')

                        # calculate additional stuff
                        add_df = combined_df2[cluster_cols + ['households']].copy()
                        clusters = add_df[cluster_cols].value_counts().reset_index()
                        clusters.columns = cluster_cols + ['clusters']
                        hh = add_df.groupby(cluster_cols)['households'].sum().reset_index()
                        hh.columns = cluster_cols + ['households sum']
                        cluster_aggregate_df = pd.merge(cluster_aggregate_df, clusters, on=cluster_cols, how='left')
                        cluster_aggregate_df = pd.merge(cluster_aggregate_df, hh, on=cluster_cols, how='left')

                        if sub_df_ind == 'All':
                            cluster_aggregate_df.to_csv(run_path + f'clustered/csv/2_clustered_{splitted_n}_{year}.csv')
                        cluster_aggregate_df = cluster_aggregate_df.reset_index()
                        if sub_df_ind == 'All':
                            cluster_aggregate_df.to_csv(run_path + f'clustered/csv/3_clustered_{splitted_n}_{year}.csv')

                        if distance in [0.5, 1, 2, 4, 6, 8, 10]:
                            combined_aggregate_l += [cluster_aggregate_df, unclustered_df, clustered_df]
                            combined_aggregate_names_l += [f"{sub_df_ind} {splitted_n} clustered {distance}km",
                                                           f"{sub_df_ind} {splitted_n} not in cluster {distance}km",
                                                           f"{sub_df_ind} {splitted_n} in cluster {distance}km"]
                            # visualizations.plot_DBSCAN(clustered_df, run_path + 'clustered/dbscan/',
                            #                            f"DBSCAN {sub_df_ind} {splitted_n} {'clustered'} {distance}km")

                        # calculate metrics
                        if f"distance" not in df_dict:
                            df_dict[f"distance"] = []
                        if sub_df_ind == 'All':
                            df_dict[f"distance"].append(distance)
                        for name, df in zip(['Clustered', 'Out of Cluster', 'In Cluster'],
                                            [cluster_aggregate_df, unclustered_df, clustered_df]):
                            if f"{sub_df_ind} {name}{year} Size" not in df_dict:
                                df_dict[f"{sub_df_ind} {name}{year} Size"] = []
                            df_dict[f"{sub_df_ind} {name}{year} Size"].append(len(df))
                            if f"{sub_df_ind} {name}{year} RMSE" not in df_dict:
                                df_dict[f"{sub_df_ind} {name}{year} RMSE"] = []
                            df_dict[f"{sub_df_ind} {name}{year} RMSE"].append(
                                ((df['Prediction'] - df['Actual']) ** 2).mean() ** .5)
                            if f"{sub_df_ind} {name}{year} Corr" not in df_dict:
                                df_dict[f"{sub_df_ind} {name}{year} Corr"] = []
                            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                            df_dict[f"{sub_df_ind} {name}{year} Corr"].append(
                                df[numeric_cols].corr()['Actual']['Prediction'])
                            # how big are differences inside of clusters (Actual and Prediction) - Can the prediction come close to the actual?
                            if name == 'In Cluster':
                                if f"Actual {sub_df_ind} {name}{year} Mean(STD)" not in df_dict:
                                    df_dict[f"Actual {sub_df_ind} {name}{year} Mean(STD)"] = []
                                df_dict[f"Actual {sub_df_ind} {name}{year} Mean(STD)"].append(
                                    df.groupby(cluster_cols)['Actual'].std().mean())
                                if f"Prediction {sub_df_ind} {name}{year} Mean(STD)" not in df_dict:
                                    df_dict[f"Prediction {sub_df_ind} {name}{year} Mean(STD)"] = []
                                df_dict[f"Prediction {sub_df_ind} {name}{year} Mean(STD)"].append(
                                    df.groupby(cluster_cols)['Prediction'].std().mean())
                            # how big are differences between clusters in different years (Actual and Prediction) - Can the prediction come close to the actual?
                            if name == 'Clustered' and splitted_n == 'location and year':
                                if f"Actual {sub_df_ind} {name}{year} Mean(STD) between years" not in df_dict:
                                    df_dict[f"Actual {sub_df_ind} {name}{year} Mean(STD) between years"] = []
                                df_dict[f"Actual {sub_df_ind} {name}{year} Mean(STD) between years"].append(
                                    df.groupby('clustered')['Actual'].std().mean())
                                if f"Prediction {sub_df_ind} {name}{year} Mean(STD) between years" not in df_dict:
                                    df_dict[f"Prediction {sub_df_ind} {name}{year} Mean(STD) between years"] = []
                                df_dict[f"Prediction {sub_df_ind} {name}{year} Mean(STD) between years"].append(
                                    df.groupby('clustered')['Prediction'].std().mean())

                # save these statistics and create plots
                stats_df = pd.DataFrame(df_dict)
                for name3 in ['Clustered', 'Out of Cluster', 'In Cluster']:
                    stats_df[f'U vs U+R size diff {name3}{year}'] = stats_df[f'U+R {name3}{year} Size'] - stats_df[
                        f'U {name3}{year} Size']
                stats_df.to_csv(os.path.join(run_path + 'clustered/', f"{splitted_n}{year} Clustered stats.csv"),
                                index=False)
                color_keywords = ['All', 'U', 'R', 'U+R']
                #, 'R+U']
                linestyle_keywords = ['Clustered', 'Out of Cluster', 'In Cluster']
                for name2 in ['Size', 'RMSE', 'Corr', 'In Cluster Mean(STD)', 'Clustered Mean(STD) between years',
                              'U vs U+R size diff']:
                    add_linestyle_to_color_keywords = True
                    if name2 == 'U vs U+R size diff':
                        color_keywords = False
                        add_linestyle_to_color_keywords = False
                    cols = [c for c in stats_df.columns if name2 in c]
                    if name2 == 'In Cluster Mean(STD)' or name2 == 'Clustered Mean(STD)':
                        linestyle_keywords = ['Actual', 'Prediction']
                    if cols:
                        visualizations.plot_dataframe(stats_df, 'distance', cols, run_path + 'clustered/',
                                                      f"{splitted_n}", name2, label_reduce=name2,
                                                      color_keywords=color_keywords,
                                                      linestyle_keywords=linestyle_keywords)

        # create scatterplots
        for name, df in zip(names + combined_aggregate_names_l,
                            dfs + combined_aggregate_l):
            # make pandas warning (setting value on copy vs view) go away
            df = df.copy()
            df['Error'] = df['Prediction'] - df['Actual']
            df['Absolute Error'] = df['Error'].abs()
            path = run_path
            if 'cluster' in name:
                path = run_path + 'clustered/scatter/'
            if 'Mozambique' in name:
                path = run_path + 'moz/'
            elif 'adm' in name:
                path = run_path + 'adm/'
            if 'different Split Models' in name:
                split_col_sdf = split_col
                sub_df = df[["Actual", "Prediction", split_col_sdf]]
            else:
                split_col_sdf = False
                sub_df = df[["Actual", "Prediction"]]
            # logger.debug('Sub df before Scatterpot %s \n%s', name, sub_df)
            if len(sub_df) > 2:
                beta, alpha, pearson_corr, rmse, nrmse = visualizations.scatterplotRegressionMultiInputs(sub_df, path,
                                                                                                         file_name=name,
                                                                                                         multidataset_col=split_col_sdf)
                if not 'cluster' in name:  # or 'clustered' in name and ('2km' in name or '3km' in name or '4km' in name or '5km' in name):
                    df.to_csv(run_path + 'csv/' + name + '.csv', index=False)
                    for na, v in zip(["beta", "alpha", "pearson_corr", "rmse", "nrmse"],
                                     [beta, alpha, pearson_corr, rmse, nrmse]):
                        eval_d[f"{name} {na}"] = v
                elif 'cluster' in name:
                    df.to_csv(run_path + 'clustered/csv/' + name + '.csv', index=False)
            if "Pred Original Data" in df.columns and cfg.report_original_data:
                if 'different Split Models' in name:
                    split_col_sdf = split_col
                    sub_df = df[["Actual Original Data", "Pred Original Data", split_col_sdf]]
                else:
                    split_col_sdf = False
                    sub_df = df[["Actual Original Data", "Pred Original Data"]]

                if len(sub_df) > 0:
                    beta, alpha, pearson_corr, rmse, nrmse = visualizations.scatterplotRegressionMultiInputs(sub_df,
                                                                                                         path,
                                                                                                         file_name=name + " Original Data",
                                                                                                         multidataset_col=split_col_sdf)
                for na, v in zip(["beta", "alpha", "pearson_corr", "rmse", "nrmse"],
                                 [beta, alpha, pearson_corr, rmse, nrmse]):
                    eval_d[f"{name} {na} Original Data"] = v
    else:
        raise NotImplementedError("Overall evaluation of a categorical model is not implemented right now")
    return eval_d


# def create_label_visualizations(run_path, labels_df, test_df, label_name, split_col_n):
#     with open(run_path + 'dfs_in', 'wb') as f:
#         pickle.dump([labels_df, test_df], f)
#     df = labels_df.copy()
#     for col in df.columns:
#         if col in ['reversed', 'transformed', 'normalized', 'label']:
#             visualizations.standard_hist_from_df(df[col], run_path, '', title_in=col)
#             if col == 'label':
#                 for split in labels_df[split_col_n].unique():
#                     sub_df = df[df[split_col_n] == split]
#                     visualizations.standard_hist_from_df(sub_df[col], run_path, col, title_in=split)
#             # visualizations.histogram(df[col], 'probability', title, run_path,
#             #                          label_name + df_n + ' ' + add_n)


def visualize_augmented_images(augmentation_d, train_ds, augmented_images_path):
    ###Visualize results
    # save augmented images
    logger.info('aug %s', augmentation_d)
    if augmentation_d:
        data_augmentation = augmentation(**augmentation_d)
    for images, _ in train_ds.take(1):
        for nr, image in enumerate(images):
            # logger.debug('%s %s', nr, image.shape)
            # logger.debug(augmentation_d)
            plt.figure(figsize=(10, 10))
            for i in range(9):
                # if augmentation_d:
                #     img = data_augmentation(image)
                img = image
                ax = plt.subplot(3, 3, i + 1)
                img = img.numpy().astype("float32")
                img = np.moveaxis(img, 0, -1)
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                plt.imshow(img)
                plt.axis("off")
            plt.savefig(augmented_images_path + str(nr) + '_unaugmented')
            plt.close()
            # plt.show()

            if augmentation_d:
                plt.figure(figsize=(10, 10))
                logger.debug('augmented image')
                for i in range(9):
                    if augmentation_d:
                        img = data_augmentation(image)
                    else:
                        img = image
                    ax = plt.subplot(3, 3, i + 1)
                    img = img.numpy().astype("float32")
                    img = np.moveaxis(img, 0, -1)
                    img = (img - np.min(img)) / (np.max(img) - np.min(img))
                    plt.imshow(img)
                    plt.axis("off")
                plt.savefig(augmented_images_path + str(nr) + '_augmented')
                plt.close()
                # plt.show()
            if nr >= cfg.save_augmented_images:
                break
    if cfg.verbose:
        logger.info('saved augmented images to %s', augmented_images_path)


def main():
    if cfg.test_mode:
        logger.warning(
            'Are you sure you want to run in test mode? Note: If using a really low amount of data defined in'
            ' test_mode, it is not guaranteed that there will be all splits and an error might occur')
        input('Press a key to continue...')
    logger.info('Using following test table (cfg.splitn_labeld_imgp_augmentation_dimension_l)')
    for i in cfg.splitn_labeld_imgp_augmentation_dimension_l:
        logger.info(i)
    # print('%s', list(cfg.splitn_labeld_imgp_augmentation_dimension_l))
    for (split_col, [label_name_in, label_d], img_path, augmentation_d,
         dim) in cfg.splitn_labeld_imgp_augmentation_dimension_l:
        label_name = label_name_in[:-1]
        logger.info('modelling water supply %s %s %s', label_name, 'type', cfg.type_m)
        # Import paths I
        prj_path = cfg.prj_folder

        if cfg.load_subfolders:
            folders = hu.files_in_folder(cfg.img_path, return_folders=True)
        else:
            folders = cfg.img_path
        logger.debug('folders %s', folders)

        logger.info('heigh, width = %s (False meaning max)', dim)
        if not dim:
            logger.warning(
                f'This model assumes that you have preprocessed images, which need to have the exact same size')
            height = False
            width = False
        else:
            height = dim
            width = dim
        if not cfg.test_mode:
            epochs = cfg.epochs
        else:
            epochs = 1
            height = 100
            width = 100
            logger.warning(f"You are in test mode! Change in config for real runs! epochs = {epochs}, image height and"
                           f"width capped to {height} {width}")

        # create new folders for run files
        train_history_path, run_path, modelcheckpoint_p, augmented_images_path, run_name = \
            create_run_folders(augmentation_d, img_path, prj_path, cfg.trainHistory_subname,
                               label_name_in, split_col, height, width, test_mode=cfg.test_mode)

        # load labels and class_weights from file (or calculate later one)
        split_dfs, test_df, labels_df, label_mapping, add_params, scaler = \
            load_labels(os.path.join(cfg.labels_f), label_name, run_path, base_path=img_path,
                        mode=cfg.type_m, split=split_col, min_year=cfg.label_min_year,
                        **label_d)
        # add_params['img height'] = height
        # add_params['img width'] = width
        # load shape
        file = split_dfs[0]['path'].iloc[0]
        arr = gu.load_geotiff(file, height=height, width=width, channel_l=cfg.channels, only_return_array=True)
        input_shape = arr.shape
        logger.info('Using following shape (must be identical for all loaded datasets) %s', input_shape)

        if cfg.type_m == 'categorical':
            num_labels = len(label_mapping)
            class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels_df['label']),
                                                              y=labels_df['label'])
            class_weights = dict(enumerate(class_weights))
        else:
            num_labels = 1
            class_weights = False

        split_dfs = [(spdf[split_col].iloc[0], spdf) for spdf in split_dfs]
        # sort by name
        split_dfs = list(sorted(split_dfs, key=lambda x: x[0]))

        run_summary_splits_f = os.path.join(train_history_path, 'run_summary_splits' + label_name_in + '.csv')
        run_summary_f = os.path.join(train_history_path, 'run_summary' + label_name_in + '.csv')
        run_summary_overall_f = os.path.join(cfg.prj_folder, cfg.trainHistory_subname, 'run_summary' + '.csv')
        if os.path.exists(run_summary_f):
            run_summary_df = pd.read_csv(run_summary_f)
        else:
            run_summary_df = False
        if os.path.exists(run_summary_overall_f):
            run_summary_overall_df = pd.read_csv(run_summary_overall_f)
        else:
            run_summary_overall_df = False

        val_dfs_l = []
        test_dfs_l = []
        add_rep = {'run_name': run_name, 'split name': split_col,
                   'Train, val, test mode': cfg.dont_use_crossval,
                   'img height': height, 'img width': width, 'label name': label_name,
                   **augmentation_d, **label_d}
        for k, v in add_rep.items():
            logger.info(f'{k}: {v}')


        test_nr = 0
        val_nr = 1
        for split_nr in enumerate(split_dfs):
            # use statistically optimal splits
            if cfg.dont_use_crossval:
                # create scores for best represented splits by means of mean and std of the data and assign test and
                # validation splits with lowest deviation of data set
                scores = hu.statistical_weighted_test_set(pd.concat([spl_df[1] for spl_df in split_dfs],
                                                                    ignore_index=True), split_col, label_name)
                test_df = scores[0][2]
                model_name = scores[0][0]
                validation_df = scores[1][2]
                val_name = scores[1][0]
                # drop validation and test splits and concat the rest to train_df
                train_dfs = [spl_df[1] for spl_df in split_dfs if spl_df[0] != val_name and spl_df[0] != model_name]
            else:
                validation_df = split_dfs[val_nr][1]
                test_df = split_dfs[test_nr][1]
                model_name = split_dfs[test_nr][0]
                val_name = split_dfs[val_nr][0]
                # drop validation and test splits and concat the rest to train_df
                train_dfs = [spl_df[1] for spl_df in split_dfs if spl_df[0] != val_name and spl_df[0] != model_name]
            train_df = pd.concat(train_dfs, ignore_index=True)

            add_rep['test split'] = model_name
            add_rep['validation split'] = val_name
            logger.info('Test mode (images) %s', cfg.test_mode)
            logger.info('input path %s', img_path)
            logger.info('Validation split %s', val_name)
            logger.info('Test split %s', model_name)
            logger.info('training on %s',
                        [spl_df[0] for spl_df in split_dfs if spl_df[0] != val_name and spl_df[0] != model_name])
            logger.debug(validation_df.head())

            # add +1 to test and val nr for next iteration
            val_nr += 1
            if val_nr == len(split_dfs):
                val_nr = 0
            test_nr += 1

            modelcheckpoint_path = modelcheckpoint_p + f'chkpt_{model_name}'
            if os.path.exists(run_summary_splits_f):
                run_summary_splits_df = pd.read_csv(run_summary_splits_f)
            else:
                run_summary_splits_df = False

            if cfg.verbose:
                if cfg.test_mode:
                    logger.debug('DS length is: %s', len(train_df))
                    logger.warning("Test mode is active! Your dataset is significantly shorter")
                # logger.info('aug_key, augmentation_d %s \n %s', aug_key, augmentation_d)
                # for spldf in split_dfs[0:split_nr] + split_dfs[split_nr + 1:]:
                # logger.debug(f'Val split {split_nr} {model_name} labels:')
                if cfg.type_m == 'categorical':
                    logger.info('class weigths', class_weights)
                    logger.info(label_mapping)
                    logger.info(labels_df['label'].value_counts())
                    logger.info(labels_df[label_name].value_counts())

            # create cache path
            cache_p = os.path.join(cfg.tmp_p, 'cache', run_name, '')
            if cfg.verbose:
                logger.debug('cache %s', cache_p)
            # ensure it gets reloaded
            if os.path.exists(cache_p):
                shutil.rmtree(cache_p)
            os.mkdir(cache_p)

            # save config
            shutil.copyfile('config.py', run_path + '/config.py')

            # strategy scope from tf.distribute.MirroredStrategy(gpus) (cf. top of file) in TF2 is used for mutlti-gpu
            # usage
            with strategy.scope():
                # Everything that creates variables should be under the strategy scope.
                # In general this is only model construction & `compile()`.
                # Load Model
                train_ds, validation_ds, test_ds, steps_per_epoch_train, steps_per_epoch_val, steps_per_epoch_test = \
                    dataset_creator(train_df, validation_df, test_df, cfg.type_m, num_labels,
                                    cache_p, input_shape, cfg.channels, **augmentation_d)

                # steps_per_epoch=False
                loss = cfg.loss()
                metrics_l = []
                for m in cfg.metrics_l:
                    metrics_l.append(m())
                if cfg.type_m == 'categorical':
                    metrics_l.append(tfa.metrics.F1Score(num_classes=num_labels,
                                                         average='micro'))
                model = model_loader(num_labels, cfg.type_m, input_shape, **augmentation_d)
                # if cfg.load_model_weights:
                #     logger.warning(f"This has not been adjusted to split stuff and only {model_name} will be"
                #                   f"evaluated")
                #     input("Are you sure you want to continue?")
                #     logger.info('Loading model from', cfg.load_model_weights + f'chkpt_{model_name}')
                #     # model = tf.keras.models.load_model(cfg.load_model_weights)
                #     model.load_weights(cfg.load_model_weights + f'chkpt_{model_name}')
                #     logger.debug('Predicting with loaded model', model)
                #     random_dic = \
                #         evaluate_dataset(model, test_ds, steps_per_epoch_test, run_path, label_mapping,
                #                         add_params, scaler, label_d, model_names, test_df, {},
                #                          add_name='_loded_model')
                #     logger.debug('returned from create_vis')
                ###Define Parameters for run
                optimizer = optimizer_loading()
                callbacks_l = callback_loader(run_path, modelcheckpoint_path, metrics_l[0])

                model.compile(optimizer=optimizer, loss=loss, metrics=metrics_l)  # , run_eagerly=True)
                # Load weights
                # buggy - don't know why (works without by name for same amount of classes/regression
                # if cfg.load_model_weights:
                #     model = model.load_weights(cfg.load_model_weights)
                #     print('loaded weights from', cfg.load_model_weights)
                logger.debug('Final Model %s', type(model))
                logger.debug(model.summary())

            ###Run Model
            t_begin = time.time()
            logger.info('steps per epoch (train/val/test) %s %s %s',
                        steps_per_epoch_train, steps_per_epoch_val, steps_per_epoch_test)
            logger.info('train shape %s', train_ds.element_spec)
            logger.info('val shape %s', validation_ds.element_spec)
            try:
                history = model.fit(
                    train_ds,
                    class_weight=class_weights,
                    validation_data=validation_ds,
                    epochs=epochs,
                    callbacks=callbacks_l,
                    steps_per_epoch=steps_per_epoch_train,
                    validation_steps=steps_per_epoch_val)
            except KeyboardInterrupt:
                history = False
                pass
            fit_time = time.time() - t_begin
            if cfg.verbose:
                logger.info('Time to fit model (s) %s', fit_time)

            # Save history as pickle
            if history:
                with open(os.path.join(run_path, 'trainHistory'), 'wb') as file_pi:
                    pickle.dump(history.history, file_pi)

            ###Visualize results
            if cfg.verbose and history:
                logger.debug(history.history.keys())
                # print(history.history)

            ###Augmented images
            if cfg.save_augmented_images:
                visualize_augmented_images(augmentation_d, train_ds, augmented_images_path)
            # delete cache_p again
            if os.path.exists(cache_p):
                shutil.rmtree(cache_p)

            # summarize statistics
            additional_reports_d = {}
            if history:
                additional_reports_d = {'max epoch': len(history.history['val_loss']),
                                        'fit time': fit_time,
                                        'Time per epoch': fit_time / len(history.history['val_loss'])}
            if history:
                for metric in metrics_l:
                    mn = metric.name
                    if cfg.type_m == 'categorical':
                        additional_reports_d['max val ' + mn] = max(history.history['val_' + mn])
                        additional_reports_d['max val epoch ' + mn] = history.history['val_' + mn].index(max(
                            history.history['val_' + mn]))
                        additional_reports_d['val ' + mn + ' at break'] = history.history['val_' + mn][-1]
                    else:
                        additional_reports_d['min val ' + mn] = min(history.history['val_' + mn])
                        additional_reports_d['min val epoch ' + mn] = history.history['val_' + mn].index(min(
                            history.history['val_' + mn]))
                        additional_reports_d['val ' + mn + ' at break'] = history.history['val_' + mn][-1]
                # visualize history
                visualizations.plot_history(history, run_path + model_name, cfg.type_m)

            ###Evaluate model
            # Evaluate the model via the test dataset
            # (highest validation accuracy seems to always perform best)
            if cfg.reload_best_weights_for_eval:
                model.load_weights(modelcheckpoint_path)

            # Save model
            model.save(os.path.join(run_path, f'Model_{model_name}'), save_format='h5')

            if cfg.verbose:
                logger.info("Evaluating on test data %s", test_ds)
            test_history = model.evaluate(test_ds, steps=steps_per_epoch_test)
            if cfg.verbose:
                logger.info("test loss, test %s %s %s", metrics_l[0].name, ':', test_history)
            for nr, r in enumerate(test_history):
                if nr == 0:
                    additional_reports_d['test_loss'] = r
                additional_reports_d['test_' + metrics_l[nr - 1].name] = r

            additional_reports_d, sk_metrics_d, validation_f_df = \
                evaluate_dataset(model, validation_ds, steps_per_epoch_val, run_path, label_mapping, add_params,
                                 scaler, label_d, model_name, validation_df, additional_reports_d,
                                 evaluate_mode='val')
            additional_reports_d, sk_metrics_d, test_f_df = \
                evaluate_dataset(model, test_ds, steps_per_epoch_test, run_path, label_mapping, add_params,
                                 scaler, label_d, model_name, test_df, additional_reports_d, label_col_in=label_name)
            val_dfs_l.append(validation_f_df)
            test_dfs_l.append(test_f_df)
            # write run_summary
            # cleaning
            if not cfg.report_original_data:
                for k, v in additional_reports_d.items():
                    if 'Original Data' in k:
                        del additional_reports_d[k]
            report_d = {**add_rep,
                        **additional_reports_d, **sk_metrics_d,
                        **add_params}
            report_df = pd.DataFrame(report_d, index=[0])
            logger.debug('split report df\n%s', report_df)
            if run_summary_splits_df is not False:
                run_summary_splits_df = pd.concat([run_summary_splits_df, report_df], axis=0, join='outer')
            else:
                run_summary_splits_df = report_df
            logger.debug('final split report\n%s', run_summary_splits_df)
            run_summary_splits_df.to_csv(run_summary_splits_f, index=False)

        ###Summary
        sub_df = run_summary_splits_df[run_summary_splits_df['run_name'] == run_name]
        mean_df = sub_df.select_dtypes(include=[np.number]).mean(axis=0)
        mean_df['split'] = 'mean'
        mean_df['run_name'] = run_name
        run_summary_splits_df = pd.concat([run_summary_splits_df, mean_df.to_frame().T], axis=0, join='outer')
        logger.debug('final run summary splits df\n%s', run_summary_splits_df)
        run_summary_splits_df.to_csv(run_summary_splits_f, index=False)

        ###Make final evaluation of all models for validation and test predictions,
        eval_d = evaluate_final_dataset(test_dfs_l, val_dfs_l, run_path, split_col=split_col,
                                        labels_df=labels_df)

        # create overall file
        del add_rep['test split']
        del add_rep['validation split']
        report_final_d = {**add_rep, **eval_d, **add_params}
        report_final_df = pd.DataFrame(report_final_d, index=[0])
        logger.info('final x-val report df\n%s', report_final_df)
        if run_summary_df is not False:
            run_summary_df = pd.concat([run_summary_df, report_final_df], axis=0, join='outer')
        else:
            run_summary_df = report_final_df
        if run_summary_overall_df is not False:
            run_summary_overall_df = pd.concat([run_summary_overall_df, report_final_df], axis=0, join='outer')
        else:
            run_summary_overall_df = report_final_df
        logger.info('final x-val run summary df\n%s', run_summary_df)
        if not cfg.test_mode or cfg.test_mode == 'test run summary':
            run_summary_df.to_csv(run_summary_f, index=False)
            run_summary_overall_df.to_csv(run_summary_overall_f, index=False)


if __name__ == "__main__":
    main()



