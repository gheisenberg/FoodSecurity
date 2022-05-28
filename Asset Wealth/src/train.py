from src.config import gpu_id
import sys

sys.path.append("..")
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
from src.data_utils import generator, create_splits
import matplotlib.pyplot as plt
from functools import partial
from operator import itemgetter

from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras import optimizers, models


from sklearn.model_selection import KFold

# import models
from src.vgg19 import VGG19_hyperspectral
from src.resnet50 import ResNet50v2_hyperspectral

import wandb
from wandb.keras import WandbCallback
try:
    from src.config import wandb_project
    from src.config import wandb_entity
    from src.config import wandb_dir
except:
    raise NameError("Set Weights and Bias Parameters in config.py first.")
import atexit

tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(len(gpus), "Physical GPU(s)")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

from src.config import img_path
from src.config import pre2015_path
from src.config import csv_path
from src.config import model_name
from src.config import k
from src.config import input_height
from src.config import input_width
from src.config import img_source
from src.config import urban_rural
from src.config import channel_size
from src.config import batch_size
from src.config import epochs
from src.config import subset
from src.config import lr
from src.config import loss
from src.config import early_stopping_patience



def main(img_dir: str, csv_path: str, pre2015_path:str, model_name: str, k: int, 
         input_height: int, input_width: int, img_source: str, urban_rural: str, 
         channel_size: int, batch_size: int, epochs: int, subset: bool):
    '''
    Train a Model with the Parameters set in config.py.
    Args:
        img_dir (str): Path to Image Data
        csv_path (str): Path to Cluster CSV Files
        pre2015_path(str):  Path to Image Data older than 2015; if all Data is used for training this shoulde be False.
        model_name (str): One of ['vgg19', 'resnet50'] to choose which Model is used
        k (int): Number of Folds for Cross Validation
        input_height (int): Pixel Height of input
        input_width (int): Pixel Width of input
        img_source (str): One of ['s2', 'viirs'] to choose whether Sentinel-2, VIIRS (nightlight) or combined Data is used
        urban_rural (str): On of ['u','r','ur'] to choose whether only urban or only rural clusters are used
        channels (list):  Channels to use; [] to use all Channels
        channel_size (int): Number of Channels (3 for RGB (VIIRS), 13 for all Sentinel2 Channels, 14 for all Channels) !Nightlight channel is transformed to 3 channels for Model Compatibility
        batch_size (int): Size of Training Batches
        epochs (int): Number of Training Epochs
        subset (bool): Whether or not to use a Subset to test the Process
    '''

    if pre2015_path: 
        X_train_val, X_test, y_train_val, y_test, X_test_pre2015, y_test_pre2015 = create_splits(img_dir, 
                                                                                                 pre2015_path,
                                                                                                 csv_path,
                                                                                                 urban_rural, 
                                                                                                 subset)
    else:
        X_train_val, X_test, y_train_val, y_test = create_splits(img_dir, 
                                                                 pre2015_path,
                                                                 csv_path,
                                                                 urban_rural, 
                                                                 subset)
    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    for fold, (train_index, val_index) in enumerate(kf.split(X_train_val)):
        print(f'Fold: {fold}')
        # Load Folds
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

        if model_name == 'vgg19':
            hyperspectral_model_obj = VGG19_hyperspectral(img_w=input_width,
                                                          img_h=input_height,
                                                          channels=channel_size
                                                          )
            model = hyperspectral_model_obj.load_vgg19()
        elif model_name == 'resnet50':
            hyperspectral_model_obj = ResNet50v2_hyperspectral(img_w=input_width,
                                                               img_h=input_height,
                                                               channels=channel_size
                                                              )
            model = hyperspectral_model_obj.load_resnet50v2()

        model.compile(optimizer=optimizers.RMSprop(learning_rate=lr),
                      loss=loss, metrics=[tf.keras.metrics.MeanSquaredError(),
                                                          tf.keras.metrics.MeanAbsoluteError(),
                                                          tf.keras.metrics.MeanAbsolutePercentageError(),
                                                          tf.keras.metrics.RootMeanSquaredError(),
                                                          tf.keras.metrics.CosineSimilarity()
                                                          ])

        wandb.init(project=wandb_project, entity=wandb_entity, dir=wandb_dir,
                   group=f'{model_name}_pretrained_model_{urban_rural}_{img_source}_{batch_size}', job_type='train',
                   name=f'{model_name}_pretrained_model_{urban_rural}_{img_source}_{batch_size}_fold_{fold}')
        config = wandb.config  # Config is a variable that holds and saves hyperparameters and inputs
        config.learning_rate = lr
        config.batch_size = batch_size
        config.epochs = epochs
        config.img_width = input_width
        config.img_height = input_height
        config.model_name = model_name
        config.pretrain_weights = 'imagenet'
        config.urban_rural = urban_rural
        config.image_source = img_dir
        config.loss = loss
        config.metrics = config.metrics = [loss,
                          'mean_absolute_error',
                          'mean_absolute_percentage_error',
                          'root_mean_squared_error',
                          'cosine_similarity']
        print('Start Model Training')
        # Fit and train model
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=1,
            callbacks=[WandbCallback(),
                       EarlyStopping(
                           monitor="val_loss",
                           patience=early_stopping_patience,
                       )])
        print(history.history)
        
        # Evaluate model for this fold.

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
        
        if pre2015_path:
            test_generator_func = partial(generator, pre2015_path, X_test_pre2015, y_test_pre2015, batch_size, input_height, input_width,
                                          channel_size)
            test_ds = Dataset.from_generator(generator=test_generator_func,
                                             output_types=(tf.float64, tf.float32),
                                             output_shapes=((batch_size, input_width, input_height, channel_size),
                                                            (batch_size,)),
                                             )
            # Evaluate on second testset
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
    main(img_dir=img_path,
         csv_path=csv_path,
         model_name=model_name,
         k=k,
         input_height=input_height,
         input_width=input_width,
         img_source=img_source,
         urban_rural=urban_rural,
         channel_size=channel_size,
         batch_size=batch_size,
         epochs=epochs,
         subset=subset)
