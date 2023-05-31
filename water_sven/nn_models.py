from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D, preprocessing
import inspect
import tensorflow as tf
from keras.applications.vgg16 import VGG16
import numpy as np

#own imports
import helper_utils as hu
import config as cfg
logger = hu.setup_logger(cfg.logging)


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
        # elif k == 'shear_range':

        else:
            raise NotImplementedError()
    logger.debug('layersl', layer_l)
    data_augmentation_layers = tf.keras.Sequential(layer_l)

    logger.debug(data_augmentation_layers)
    return data_augmentation_layers


def add_classification_top_layer(model, out_classes, neurons_l, type_m='categorical', dropout=0.5, unfreeze_layers=0):
    """Adds custom top layers

    Args:
        model (keras model): Pass the base model
        out_classes (int): Define how many classes the classification layer shall have
        neurons_l (list): Define how many hidden layers the top model shall have (len(neurons_l)) and how many neurons
            these layers shall have (ints)
        dropout (float): Define the dropout in these hidden layers
        unfreeze_layers (int): Percentage of how many layers shall be learnable/unfrozen

    Returns:
        model (keras model): Final model
    """
    if unfreeze_layers:
        #freeze layers of input model
        #unfreeze at least one layer if unfreeze layers != 0
        unfrozen_layers = max(1, round(len(model.layers) * unfreeze_layers/100))
        freeze_layers = len(model.layers) - unfrozen_layers
        for layer in model.layers[0:freeze_layers]:
            layer.trainable = False
        logger.debug('Frozen layers %s %s %s %s %s', freeze_layers, 'unfrozen layers', unfrozen_layers,
                     'ges_layers', len(model.layers))
    #Add extra layers and always pass the output tensor to next layer
    x = model.output
    #to do: try out different pooling layers
    x = GlobalAveragePooling2D()(x)
    #add multiple layers defined in neurons_l
    for neurons in neurons_l:
        #named randomly to prevent reloading of weights with model.load_weights(by_name=True)
        x = Dense(neurons, activation='relu')(x)
        if dropout:
            x = Dropout(dropout)(x)
    #add softmax for
    if type_m == 'categorical':
        out = Dense(out_classes, activation='softmax')(x)
    elif type_m == 'regression':
        out = Dense(1, kernel_initializer='normal')(x)

    model = tf.keras.models.Model(inputs=model.input, outputs=out)
    return model

