import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D

import numpy as np


class VVG19_hyperspectral():
    def __init__(self,
                 img_w: int,
                 img_h: int,
                 channels: int):
        """

        Args:
            img_w: Pixel width for input images
            img_h: Pixel height for input images
            channels: number of channels for input images
        """
        self.img_w = img_w
        self.img_h = img_h
        self.channels = channels

        self.vgg19 = VGG19(weights='imagenet', include_top=False,
                           input_tensor=Input(shape=(self.img_h, self.img_w, 3), name='input_sentinel'))

        # Load part of the VGG without the top layers into 'pretrained' model
        self.pretrained_model = models.Model(inputs=self.vgg19.input,
                                             outputs=self.vgg19.get_layer('block5_pool').output)
        self.config = self.pretrained_model.get_config()
        if self.channels == 1:
            self.hs_inputs = Input(shape=(self.img_h, self.img_w, 3), name='input')
        else:
            self.hs_inputs = Input(shape=(self.img_h, self.img_w, self.channels), name='input')

    def load_vgg19(self):
        """

         Returns: Create a Model Template of VGG19 with hyperspectral input shape (as defined in init)

         """

        ## adapted from https://stackoverflow.com/questions/53251827/pretrained-tensorflow-model-rgb-rgby-channel-extension
        ## set up vgg19 template with hyperspectral input vector

        x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block1_conv1')(
            self.hs_inputs)
        x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block1_conv2')(x)
        x = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(x)

        # block 2
        x = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block2_conv2')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool')(x)

        # block 3
        x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block3_conv3')(x)
        x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block3_conv4')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool')(x)

        # block 4
        x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block4_conv3')(x)
        x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block4_conv4')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool')(x)

        # block 5
        x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block5_conv3')(x)
        x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='zeros', name='block5_conv4')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block5_pool')(x)

        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(1)(x)

        vgg_template = models.Model(inputs=self.hs_inputs, outputs=x)

        if self.channels > 3:

            layers_to_modify = ['block1_conv1']  # Turns out the only layer that changes
            # shape due to 4th to 13th channel is the first
            # convolution layer.
            for layer in self.pretrained_model.layers:  # pretrained Model and template have the same
                # layers, so it doesn't matter which to
                # iterate over.

                if layer.get_weights():  # Skip input, pooling and no weights layers

                    target_layer = vgg_template.get_layer(name=layer.name)
                    # print(target_layer.get_weights())
                    if layer.name in layers_to_modify:

                        kernels = layer.get_weights()[0]
                        biases = layer.get_weights()[1]
                        kernels_extra_channel = np.concatenate((kernels,
                                                                kernels[:, :, -3:, :],
                                                                kernels[:, :, -3:, :],
                                                                kernels[:, :, -3:, :],
                                                                kernels[:, :, -1:, :]),
                                                               axis=-2)  # For channels_last

                        target_layer.set_weights([kernels_extra_channel, biases])

                    else:
                        target_layer.set_weights(layer.get_weights())

        return vgg_template