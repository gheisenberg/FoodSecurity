import keras
import tensorflow.python.keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D


import numpy as np

class VVG19_hyperspectral():
    def __init__(self,
                 img_w: int,
                 img_h: int,
                 channels: int):
        '''

        Args:
            img_w: Pixel width for input images
            img_h: Pixel height for input images
            channels: number of channels for input images
        '''
        self.img_w = img_w
        self.img_h = img_h
        self.channels = channels

        self.inputs = layers.Input(shape=(self.img_w, self.img_h, self.channels), name='input_sentinel')

        self.vgg19 = VGG19(weights='imagenet')

        self.pretrained_model = models.Model(inputs=self.vgg19.input, outputs=self.vgg19.get_layer('block5_pool').output)
        self.config = self.pretrained_model.get_config()


    def load_vgg19_hyperspectral_template(self):
        '''

        Returns: Create a Model Template of VGG19 with hyperspectral input shape (as defined in init)

        '''
        ## set up vgg19 template with hyperspectral input vector
        x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block1_conv1')(
            self.inputs)
        for layer in self.config['layers'][2:]:
            print(layer)
            if layer['class_name'] == 'Conv2D':
                x = Conv2D(layer['config']['filters'], layer['config']['kernel_size'],
                           padding=layer['config']['padding'], activation=layer['config']['activation'],
                           kernel_initializer=layer['config']['kernel_initializer']['class_name'], name=layer['name'])(
                    x)
            elif layer['class_name'] == 'MaxPooling2D':
                x = MaxPooling2D(pool_size=layer['config']['pool_size'], name=layer['name'])(x)

        x = layers.Flatten()(x)
        x = layers.Dense(128)(x)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1)(x)
        x = layers.Activation("softmax", name="asset_wealth")(x)

        vgg_template = models.Model(inputs=self.inputs, outputs=x)
        print(vgg_template.summary())

        return vgg_template

    ## adapted from https://stackoverflow.com/questions/53251827/pretrained-tensorflow-model-rgb-rgby-channel-extension
    def resize_weights(self, vgg_template:keras.Model):
        '''

        Args:
            vgg_template: Template Model with hyperspectral input size

        Returns:
            Keras Model with VGG19 weights pretrained on ImageNet but with Hyperspectral input size
        '''
        for pretrained_layer, layer in zip(self.pretrained_model.layers, vgg_template.layers):
            pretrained = self.pretrained_model.get_weights()
            target = layer.get_weights()
            if len(pretrained) == 0:  # skip input, pooling and other no weights layers
                continue
            try:
                # set the pretrained weights as is whenever possible
                layer.set_weights(pretrained)
            except:
                # numpy.resize to the rescue whenever there is a shape mismatch
                for idx, (l1, l2) in enumerate(zip(pretrained, target)):
                    target[idx] = np.resize(l1, l2.shape)

                layer.set_weights(target)
        return vgg_template