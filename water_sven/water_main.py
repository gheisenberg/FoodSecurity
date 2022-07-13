#!/usr/bin/env python
# coding: utf-8

###general imports
import os
import shutil
import pandas as pd
import numpy as np
import tensorflow
from sklearn.metrics import classification_report
import keras
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
import pickle
import time
import csv
from functools import partial
#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()
# warnings.filterwarnings("ignore")
# tf.autograph.set_verbosity(1)
import sys

###priv imports
import config as cfg
import visualizations
import nn_utils as nnu
import nn_models as nnm
import helper_utils as hu

###Some general information and settings
#print some general information
print('keras v', keras.__version__)
print('tf keras v', tensorflow.keras.__version__)
print('tf v', tf.__version__)
#to do: try non eager execution graph?
print('tf eager execution', tf.executing_eagerly())

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
print('gpus', gpus)
strategy = tf.distribute.MirroredStrategy(gpus)
print('Number of GPUs MirroredStrategy: {}'.format(strategy.num_replicas_in_sync))


def create_run_folders(augmentation_d, normalization_d, prj_path, trainHistory_subname):
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
    aug_name = '_'
    for k, v in {**augmentation_d, **normalization_d}.items():
        if v is False or v == 0 or v == {}:
            pass
        elif v is True:
            aug_name += k + '_'
        else:
            aug_name += k + str(v)
    #replace some special chars and omit some strings for better readability and acceptable length
    aug_name = aug_name.replace(',', '_')
    aug_name = aug_name.replace('range', '')
    aug_name = aug_name.replace('_center_featurewise_std_normalization_', '')
    aug_name = aug_name.replace('_center_samplewise_std_normalization_', '')
    # strip every char containing harmful chars for paths
    aug_name = aug_name.translate({ord(c): None for c in '[],;!@#$ '})
    
    #create base name
    run_name = cfg.model_name + cfg.run_name_custom_string + '_w' + \
                           str(cfg.cnn_settings_d['weights']) + '_unfl' + str(cfg.unfreeze_layers_perc) + \
                           '_d' + str(cfg.dropout_top_layers) + '_lr' + str(cfg.lr)
    if cfg.auto_adjust_lr[0]:
        run_name += '_adjustlr'
    if cfg.momentum:
        run_name += '_momentum' + str(cfg.momentum)
    run_name += '_optimizer' + cfg.optimizer + aug_name

    # Create paths II
    paths_dict = {'train_history_path': ['base_path', trainHistory_subname, False],
                  'run_path': ['train_history_path', run_name, True],
                  'modelcheckpoint_path': ['run_path', 'modelcheckp/', False],
                  'augmented_imgs_path': ['run_path', 'augmented_imgs/', False]}
    [train_history_path, run_path, modelcheckpoint_path, augmented_images_path] = \
        hu.paths_from_base_path(prj_path, paths_dict)
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


def callback_loader(run_path, modelcheckpoint_path):
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
            monitor='val_' + tf.keras.metrics.CategoricalAccuracy().name, factor=cfg.auto_adjust_lr[2],
            patience=cfg.auto_adjust_lr[1], mode='auto')
        callbacks_l.append(reduce)
    if cfg.early_stopping[0]:
        # Stops the training process if the validation loss stops improving in some epochs (here 10)
        early = callbacks.EarlyStopping(monitor='val_' + tf.keras.metrics.CategoricalAccuracy().name,
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
                                                 monitor='val_' + tf.keras.metrics.CategoricalAccuracy().name,
                                                 verbose=1,
                                                 save_weights_only=True,
                                                 save_best_only=True))
    return callbacks_l


def model_loader():
    """Summarizing model loading from string defined in config.py to simplify code in main
    Also loads lots of variables from config.py, especially: cnn_settings_d (and many more)
    If defined in config.py add_custom_top_layers it will also add custom top layers by calling add_top_layer from
    nn_models.py!

    Returns:
        model (Keras model): e.g. VGG19 or resnet50 if defined with custom top layers
    """
    if cfg.model_name == 'resnet152':
        base_model = tf.keras.applications.ResNet152V2(**cfg.cnn_settings_d)
    elif cfg.model_name == 'resnet50':
        base_model = tf.keras.applications.ResNet50V2(**cfg.cnn_settings_d)
    elif cfg.model_name == 'vgg16':
        base_model = tf.keras.applications.VGG16(**cfg.cnn_settings_d)
    elif cfg.model_name == 'vgg19':
        base_model = tf.keras.applications.VGG19(**cfg.cnn_settings_d)
    elif cfg.model_name == 'inceptionv3':
        base_model = tf.keras.applications.InceptionV3(**cfg.cnn_settings_d)
    elif cfg.model_name == 'densenet121':
        base_model = tf.keras.applications.DenseNet121(**cfg.cnn_settings_d)
    elif cfg.model_name == 'densenet201':
        base_model = tf.keras.applications.DenseNet201(**cfg.cnn_settings_d)
    elif cfg.model_name == 'xception':
        base_model = tf.keras.applications.Xception(**cfg.cnn_settings_d)
    else:
        raise NotImplementedError("The model you want is not implemented right now", cfg.model_name)

    # Add custom Top Layers (Imagenet has 1000nds of classes - we need 3!)
    if cfg.cnn_settings_d['include_top'] is False and cfg.add_custom_top_layers:
        base_model = nnm.add_classification_top_layer(base_model, cfg.cnn_settings_d['classes'], cfg.neurons_l,
                                                      cfg.dropout_top_layers, cfg.unfreeze_layers_perc)
        print('Final Model with added top layers', type(base_model), 'layers', len(base_model.layers),
              'of which', round(len(base_model.layers) * cfg.unfreeze_layers_perc / 100), 'are unfroozen or ',
              cfg.unfreeze_layers_perc, '%')
    return base_model


def generator_n_dataset_creator(train_path, val_path, test_path, labels_df):
    """Summarizing generator creation to simplify code in main
    Also loads lots of variables from config.py

    Args:
        train_path, val_path, test_path (str): Paths...
        labels_df (Pandas DF): Labels

    Returns:
        train_ds, val_ds, test_ds (TF dataset): ...
    """
    # create generators and datasets (ds)
    # test if possible before loops!
    train_generator_func = partial(nnu.generator, train_path, labels_df,
                                   cfg.batch_size, cfg.input_shape[0][1],
                                   cfg.input_shape[0][2], cfg.clipping_values, cfg.channels, cfg.channel_size,
                                   cfg.num_labels)

    train_ds = tf.data.Dataset.from_generator(train_generator_func,
                                              output_types=(tf.float32, tf.float32),
                                              output_shapes=cfg.input_shape, )

    # This part generates the actual validation generator for the NN
    val_generator_func = partial(nnu.generator, val_path, labels_df,
                                 cfg.batch_size, cfg.input_shape[0][1],
                                 cfg.input_shape[0][2], cfg.clipping_values, cfg.channels, cfg.channel_size,
                                 cfg.num_labels)
    val_ds = tf.data.Dataset.from_generator(val_generator_func,
                                            output_types=(tf.float32, tf.float32),
                                            output_shapes=cfg.input_shape, )
    test_generator_func = partial(nnu.generator, test_path, labels_df,
                                  cfg.batch_size, cfg.input_shape[0][1],
                                  cfg.input_shape[0][2], cfg.clipping_values, cfg.channels, cfg.channel_size,
                                  cfg.num_labels)
    test_ds = tf.data.Dataset.from_generator(test_generator_func,
                                             output_types=(tf.float32, tf.float32),
                                             output_shapes=((cfg.batch_size, 200, 200, 3),
                                                            (cfg.batch_size, 3)), )
    return train_ds, val_ds, test_ds


def IDG_creator(train_x, train_y, val_x, val_y, test_x, test_y, augmentation_d, normalization_d):
    """Summarizing ImageDataGenerator (IDG) creation to simplify code in main
    Also loads lots of variables from config.py

    Args:
        train_x, train_y, val_x, val_y, test_x, test_y (np array): ...
        normalization_d (dict): Settings for normalization
        augmentation_d (dict): Settings for augmentation

    Returns:
        datagen_train, datagen_val, datagen_test (Keras ImageDataGenerator): ...
    """
    if cfg.verbose:
        print('Shapes trainX, trainY, valX, valY, testX, testY', train_x.shape, train_y.shape, val_x.shape, val_y.shape,
              test_x.shape, test_y.shape)
    # merge options and feed them to IDG
    datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(**{**augmentation_d, **normalization_d})
    # create data gen for val data and dont augment, so real data is evaluated and loaded from memory
    # only use augmentation for val data if it changes values - needs to be done for test data as
    # well then...
    # only use fitting of IDG when necessary
    if 'featurewise_center' in normalization_d or \
            'featurewise_std_normalization' in normalization_d or \
            'zca_whitening' in normalization_d:
        print("Fitting datagen")
        # to do fit and then add further options
        # to do zca on val
        datagen2 = tf.keras.preprocessing.image.ImageDataGenerator(**normalization_d)
        if 'zca_whitening' in normalization_d:
            # possible to only fit on small subset (and val ds is practically a subset)
            # to do: replace with subset of train
            datagen_train.fit(train_x[:int(len(train_x) * cfg.zca_whitening_perc_fit / 100)])
            datagen2.fit(train_x[:int(len(train_x) * cfg.zca_whitening_perc_fit / 100)])
        elif 'featurewise_center' in normalization_d or \
                'featurewise_std_normalization' in normalization_d:
            datagen_train.fit(train_x)
            datagen2.fit(train_x)
        # use the same IDG for val and test if fitted
        # to do: implement for test
        datagen_test = datagen2
        datagen_val = datagen2
        print('Using normalized datagen for validation and testing')
    else:
        print('Using empty IDG/no augmentation for validation and testing')
        datagen_test = tf.keras.preprocessing.image.ImageDataGenerator()
        datagen_val = datagen_test
        print(datagen_val)
    return datagen_train, datagen_val, datagen_test


def main():
    print('modelling water supply')
    # Import paths I
    prj_path = cfg.prj_folder
    paths_dict = {
                  'sentinel_path': ['base_path', 'Sentinel2/', False],
                  'sentinel_img_path': ['sentinel_path', 'preprocessed/water/urban/', False],
                  'train_path': ['sentinel_img_path', 'train/', False],
                  'val_path': ['sentinel_img_path', 'validation/', False],
                  'test_path': ['sentinel_img_path', 'test/', False]}
    # create paths
    [sentinel_path, sentinel_img_path, train_path, val_path, test_path] = \
        hu.paths_from_base_path(cfg.base_folder, paths_dict)

    #load labels and class_weights from file (or calculate later one)
    labels_df = pd.read_csv(os.path.join(prj_path, 'labels_df_shannon.csv'))
    if cfg.verbose:
        print('labels')
        print(labels_df.head())
    class_weights = hu.load_class_weights(os.path.join(prj_path, 'class_weights'), labels_df, train_path, prj_path,
                                          True, verbose=1)

    #iterating over multiple normalization and augmentation Settings of IDG (Test routine)
    for normalization_key, normalization_d in cfg.IDG_normalization_d.items():
        t_begin = time.time()
        with strategy.scope():
            train_ds, val_ds, test_ds = generator_n_dataset_creator(train_path, val_path, test_path, labels_df)
        if cfg.generator == 'ImageDataGenerator':
            with strategy.scope():
                [(train_x, train_y), (val_x, val_y), (test_x, test_y)], t_ele, t_transform, t_ges_transform = \
                    nnu.transform_data_for_ImageDataGenerator([train_ds, val_ds, test_ds])
        load_time = time.time() - t_begin
        if cfg.verbose:
            print('Loaded data in (s)', load_time)
        for aug_key, augmentation_d in cfg.IDG_augmentation_settings_d.items():
            if cfg.verbose:
                print('normalization_key', normalization_key)
                print('normalization_d', normalization_d)
                print('aug_key, augmentation_d', aug_key, augmentation_d)

            #create new folders for run files
            train_history_path, run_path, modelcheckpoint_path, augmented_images_path, run_name = \
                create_run_folders(augmentation_d, normalization_d, prj_path, cfg.trainHistory_subname)

            # save config
            shutil.copyfile('config.py', run_path + '/config.py')

            ###Define Parameters for run
            optimizer = optimizer_loading()
            callbacks_l = callback_loader(run_path, modelcheckpoint_path)

            #strategy scope from tf.distribute.MirroredStrategy(gpus) (cf. top of file) in TF2 is used for mutlti-gpu
            # usage
            with strategy.scope():
                # Everything that creates variables should be under the strategy scope.
                # In general this is only model construction & `compile()`.
                # Load Model
                model = model_loader()
                # Create Metrics
                metrics_l = [tf.keras.metrics.CategoricalAccuracy()]
                model.compile(optimizer=optimizer, loss=cfg.loss, metrics=[metrics_l])
                # Load weights
                if cfg.load_model_weights:
                    model.load_weights(cfg.load_model_weights)
                if cfg.verbose:
                    print('Final Model', type(model))
                    print(model.summary())

            if cfg.generator == 'ImageDataGenerator':
                datagen_train, datagen_val, datagen_test = IDG_creator(train_x, train_y, val_x, val_y, test_x, test_y,
                                                                       augmentation_d, normalization_d)

            ###Run Model
            t_begin = time.time()
            # to do: turn of cryptic warning (no influences though?) - not working w IDG - needs to be fixed!
            # cf. https://stackoverflow.com/questions/65322700/tensorflow-keras-consider-either-turning-off-auto-sharding-or-switching-the-a
            # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
            # use option autoencoder on ds
            # options = tf.data.Options()
            # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            # train_ds = train_ds.with_options(options)
            # val_ds = val_ds.with_options(options)
            # test_ds = test_ds.with_options(options)
            history = model.fit(
                                datagen_train.flow(train_x, train_y, batch_size=cfg.batch_size, shuffle=True),
                                class_weight=class_weights,
                                validation_data=datagen_val.flow(val_x, val_y,
                                                                 batch_size=cfg.batch_size, shuffle=True),
                                #datagen.flow_from_dataframe(val_ds),
                                epochs=cfg.epochs,
                                callbacks=callbacks_l)
            fit_time = time.time() - t_begin
            if cfg.verbose:
                print('Time to fit model (s)', fit_time)

            # Save history as pickle
            with open(os.path.join(run_path, 'trainHistory'), 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
            # Save model
            model.save(os.path.join(run_path, 'Model'))


            ###Visualize results
            # show and/or save augmented images
            if cfg.save_augmented_images:
                for datag, (x, y), add_name_img in zip([datagen_train, datagen_val, datagen_test],
                                                       [(train_x, train_y), (val_x, val_y), (test_x, test_y)],
                                                       ['train', 'val', 'test']):
                    datagen = datag.flow(x, y, batch_size=1, save_to_dir=augmented_images_path,
                                         save_prefix=add_name_img)
                    for nr1, i in enumerate(datagen):
                        if nr1 >= cfg.save_augmented_images:
                            break
                if cfg.verbose:
                    print('saved augmented images to', augmented_images_path)

            if cfg.verbose:
                print(history.history.keys())
            #summarize statistics
            additional_reports_d = {'max epoch': len(history.history['val_loss']),
                                    'max val acc': max(history.history['val_categorical_accuracy']),
                                    'max val acc epoch':
                                        history.history['val_categorical_accuracy'].index(max(
                                            history.history['val_categorical_accuracy'])),
                                    'val acc at break': history.history['val_categorical_accuracy'][-1],
                                    'fit time': fit_time, 'Time per epoch': fit_time / len(history.history['val_loss'])}
            #visualize history
            visualizations.plot_history(history, run_path)

            ### Evaluation
            # Evaluate the model via the test dataset
            # (highest validation accuracy seems to always perform best)
            if cfg.reload_best_weights_for_eval:
                model.load_weights(modelcheckpoint_path)
            else:
                break
            if cfg.verbose:
                print("Evaluate on test data")
            results = model.evaluate(datagen_test.flow(test_x, test_y))
            if cfg.verbose:
                print("test loss, test acc:", results)
            additional_reports_d['test_acc'] = results[1]

            # Create Confusion matrix for the test dataset
            # Get np array of predicted labels and true labels for test dataset
            test_true_list = []
            test_pred = []
            batches = 0
            for j in datagen_test.flow(test_x, test_y, batch_size=int(cfg.batch_size)):
                batches += 1
                ynew = model.predict(j[0])
                pred = np.argmax(ynew, axis=1)
                for ele in pred:
                    test_pred.append(ele)
                test_gold = np.argmax(j[1], axis=1)
                for ele in test_gold:
                    test_true_list.append(ele)
                if batches >= len(test_x) / int(cfg.batch_size):
                    break

            test_prediction = np.array(test_pred)
            test_true = np.array(test_true_list)

            # Confusion matrix
            cm_plot_labels = ['piped water', 'groundwater', 'bottled water']
            visualizations.plot_CM(test_true, test_prediction, cm_plot_labels, run_path +
                                   'ConfusionMatrix')
            # create sklearn report (for f1 score and more)
            classification_d = classification_report(test_true, test_prediction, output_dict=True,
                                                     target_names=cm_plot_labels)
            print('test classification', classification_d)
            sk_metrics_d = {}
            # transform for one row
            for class_name, dic in classification_d.items():
                try:
                    for metric_n, v in dic.items():
                        sk_metrics_d[class_name + ': ' + metric_n] = v
                except AttributeError:
                    sk_metrics_d[class_name] = dic

            # write report (it's a bit messy right now!)
            report_d = {**additional_reports_d, **sk_metrics_d, **{'run_name': run_name}}
            filename = os.path.join(train_history_path, 'run_summary.csv')
            file_exists = os.path.isfile(filename)
            # append to file
            with open(filename, 'a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=report_d.keys())
                if not file_exists:
                    writer.writeheader()  # file doesn't exist yet, write a header
                for k, v in report_d.items():
                    print(k, v)
                writer.writerow(report_d)


if __name__ == "__main__":
    main()



