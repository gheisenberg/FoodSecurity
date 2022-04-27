import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Reshape
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout
from keras.layers.merge import add
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from tensorflow.keras.applications.resnet import ResNet50


import numpy as np


class ResNet50_hyperspectral():
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

        self.resnet50 = ResNet50(weights='imagenet', include_top=False,
                                 input_tensor=layers.Input(shape=(self.img_h, self.img_w, 3), name='input_sentinel'))
        # Load part of the VGG without the top layers into 'pretrained' model
        self.pretrained_model = models.Model(inputs=self.resnet50.input,
                                             outputs=self.resnet50.get_layer('conv5_block3_out').output)
        self.config = self.pretrained_model.get_config()

        if self.channels == 1:
            self.hs_inputs = Input(shape=(self.img_h, self.img_w, 3), name='input')
        else:
            self.hs_inputs = Input(shape=(self.img_h, self.img_w, self.channels), name='input')

    # adapted from https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/applications/resnet.py to fit
    # hyperspectral input images
    def load_resnet50(self):
        def _shortcut(input_feature, residual, conv_name_base=None, bn_name_base=None):
            """Adds a shortcut between input and residual block and merges them with "sum"
            """
            # Expand channels of shortcut to match residual.
            # Stride appropriately to match residual (width, height)
            # Should be int if network architecture is correctly configured.
            input_shape = K.int_shape(input_feature)
            residual_shape = K.int_shape(residual)
            stride_width = int(round(input_shape[-3] / residual_shape[-3]))
            stride_height = int(round(input_shape[-2] / residual_shape[-2]))
            equal_channels = input_shape[-1] == residual_shape[-1]

            shortcut = input_feature
            # 1 X 1 conv if shape is different. Else identity.
            if stride_width > 1 or stride_height > 1 or not equal_channels:
                print('reshaping via a convolution...')
                if conv_name_base is not None:
                    conv_name_base = conv_name_base + '1'
                shortcut = Conv2D(filters=residual_shape[-1],
                                  kernel_size=(1, 1),
                                  strides=(stride_width, stride_height),
                                  padding="valid",
                                  kernel_initializer="he_normal",
                                  kernel_regularizer=l2(0.0001),
                                  name=conv_name_base)(input_feature)
                if bn_name_base is not None:
                    bn_name_base = bn_name_base + '1'
                shortcut = BatchNormalization(axis=-1,
                                              name=bn_name_base)(shortcut)

            return add([shortcut, residual])
        def _bn_relu(x, bn_name=None, relu_name=None):
            """Helper to build a BN -> relu block
            """
            norm = BatchNormalization(axis=-1, name=bn_name)(x)
            return Activation("relu", name=relu_name)(norm)

        def _conv_bn_relu(**conv_params):
            """Helper to build a conv -> BN -> relu residual unit activation function.
               This is the original ResNet v1 scheme in https://arxiv.org/abs/1512.03385
            """
            filters = conv_params["filters"]
            kernel_size = conv_params["kernel_size"]
            strides = conv_params.setdefault("strides", (1, 1))
            dilation_rate = conv_params.setdefault("dilation_rate", (1, 1))
            conv_name = conv_params.setdefault("conv_name", None)
            bn_name = conv_params.setdefault("bn_name", None)
            relu_name = conv_params.setdefault("relu_name", None)
            kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
            padding = conv_params.setdefault("padding", "same")
            kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

            def f(x):
                x = Conv2D(filters=filters, kernel_size=kernel_size,
                           strides=strides, padding=padding,
                           dilation_rate=dilation_rate,
                           kernel_initializer=kernel_initializer,
                           kernel_regularizer=kernel_regularizer,
                           name=conv_name)(x)
                return _bn_relu(x, bn_name=bn_name, relu_name=relu_name)

            return f

        def _bn_relu_conv(**conv_params):
            """Helper to build a BN -> relu -> conv residual unit with full pre-activation
            function. This is the ResNet v2 scheme proposed in
            http://arxiv.org/pdf/1603.05027v2.pdf
            """
            filters = conv_params["filters"]
            kernel_size = conv_params["kernel_size"]
            strides = conv_params.setdefault("strides", (1, 1))
            dilation_rate = conv_params.setdefault("dilation_rate", (1, 1))
            conv_name = conv_params.setdefault("conv_name", None)
            bn_name = conv_params.setdefault("bn_name", None)
            relu_name = conv_params.setdefault("relu_name", None)
            kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
            padding = conv_params.setdefault("padding", "same")
            kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

            def f(x):
                activation = _bn_relu(x, bn_name=bn_name, relu_name=relu_name)
                return Conv2D(filters=filters, kernel_size=kernel_size,
                              strides=strides, padding=padding,
                              dilation_rate=dilation_rate,
                              kernel_initializer=kernel_initializer,
                              kernel_regularizer=kernel_regularizer,
                              name=conv_name)(activation)

            return f

        def _residual_block(block_function, filters, blocks, stage,
                            transition_strides=None, transition_dilation_rates=None,
                            dilation_rates=None, is_first_layer=False, dropout=None,
                            residual_unit=_bn_relu_conv):
            """Builds a residual block with repeating bottleneck blocks.
               stage: integer, current stage label, used for generating layer names
               blocks: number of blocks 'a','b'..., current block label, used for generating
                    layer names
               transition_strides: a list of tuples for the strides of each transition
               transition_dilation_rates: a list of tuples for the dilation rate of each
                    transition
            """
            if transition_dilation_rates is None:
                transition_dilation_rates = [(1, 1)] * blocks
            if transition_strides is None:
                transition_strides = [(1, 1)] * blocks
            if dilation_rates is None:
                dilation_rates = [1] * blocks

            def f(x):
                for i in range(blocks):
                    is_first_block = is_first_layer and i == 0
                    x = block_function(filters=filters, stage=stage, block=i,
                                       transition_strides=transition_strides[i],
                                       dilation_rate=dilation_rates[i],
                                       is_first_block_of_first_layer=is_first_block,
                                       dropout=dropout,
                                       residual_unit=residual_unit)(x)
                return x

            return f

        def _block_name_base(stage, block):
            """Get the convolution name base and batch normalization name base defined by
            stage and block.
            If there are less than 26 blocks they will be labeled 'a', 'b', 'c' to match the
            paper and keras and beyond 26 blocks they will simply be numbered.
            """
            if block < 27:
                block = '%c' % (block + 97)  # 97 is the ascii number for lowercase 'a'
            conv_name_base = 'res' + str(stage) + block + '_branch'
            bn_name_base = 'bn' + str(stage) + block + '_branch'
            return conv_name_base, bn_name_base

        def bottleneck(filters, stage, block, transition_strides=(1, 1),
                       dilation_rate=(1, 1), is_first_block_of_first_layer=False, dropout=None,
                       residual_unit=_bn_relu_conv):
            """Bottleneck architecture for > 34 layer resnet.
            Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
            Returns:
                A final conv layer of filters * 4
            """

            def f(input_feature):
                # don't repeat bn->relu since we just did bn->relu->maxpool
                conv_name_base, bn_name_base = _block_name_base(stage, block)
                if is_first_block_of_first_layer:
                    x = Conv2D(filters=filters, kernel_size=(1, 1),
                                   strides=transition_strides,
                                   dilation_rate=dilation_rate,
                                   padding="same",
                                   kernel_initializer="he_normal",
                                   kernel_regularizer=l2(1e-4),
                                   name=conv_name_base + '2a')(input_feature)
                else:
                    x = residual_unit(filters=filters, kernel_size=(1, 1),
                                      strides=transition_strides,
                                      dilation_rate=dilation_rate,
                                      conv_name_base=conv_name_base + '2a',
                                      bn_name_base=bn_name_base + '2a')(input_feature)

                if dropout is not None:
                    x = Dropout(dropout)(x)

                x = residual_unit(filters=filters, kernel_size=(3, 3),
                                  conv_name_base=conv_name_base + '2b',
                                  bn_name_base=bn_name_base + '2b')(x)

                if dropout is not None:
                    x = Dropout(dropout)(x)

                x = residual_unit(filters=filters * 4, kernel_size=(1, 1),
                                  conv_name_base=conv_name_base + '2c',
                                  bn_name_base=bn_name_base + '2c')(x)

                return _shortcut(input_feature, x)

            return f

        def do_dilated_conv(indicies, rate, filters_out, filters_in, name):
            return Conv2D(
                filters=round(round(filters_out / filters_in, 1) * len(indicies), 0),
                kernel_size=ksize,
                padding='same',
                dilation_rate=rate,
                activation='relu',
                kernel_initializer='zeros',
                input_shape=split_x[indicies[0]].shape,
                name=name)(tf.concat(axis=-1, values=[band for i, band in enumerate(split_x)
                                                      if i in indicies]))

        activation = 'softmax'
        repetitions = [3, 4, 6, 3]
        initial_filters = 64
        initial_strides = (2, 2)
        initial_kernel_size = (7, 7)
        initial_pooling = 'max'
        transition_dilation_rate = (1, 1)
        dropout = None
        block = 'bottleneck'


        block_fn = bottleneck
        residual_unit = _bn_relu_conv

        x = _conv_bn_relu(filters=initial_filters, kernel_size=initial_kernel_size,
                              strides=initial_strides)(self.hs_inputs)
        x = MaxPooling2D(pool_size=(3, 3), strides=initial_strides, padding="same")(x)

        block = x
        filters = initial_filters

        for i, r in enumerate(repetitions):
            transition_dilation_rates = [transition_dilation_rate] * r
            transition_strides = [(1, 1)] * r
            if transition_dilation_rate == (1, 1):
                transition_strides[0] = (2, 2)
            block = _residual_block(block_fn, filters=filters,
                                    stage=i, blocks=r,
                                    is_first_layer=(i == 0),
                                    dropout=dropout,
                                    transition_dilation_rates=transition_dilation_rates,
                                    transition_strides=transition_strides,
                                    residual_unit=residual_unit)(block)
            filters *= 2

        # Last activation
        x = _bn_relu(block)

        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(1)(x)
        resnet50_template = Model(inputs=self.hs_inputs, outputs=x)

        if self.channels > 3:

            layers_to_modify = ['conv2d']  # Turns out the only layer that changes
            # shape due to 4th to 13th channel is the first
            # convolution layer.
            for layer in resnet50_template.layers:  # pretrained Model and template have the same
                # layers, so it doesn't matter which to
                # iterate over.

                if layer.get_weights():  # Skip input, pooling and no weights layers

                    target_layer = resnet50_template.get_layer(name=layer.name)
                    # print(target_layer.get_weights())
                    if layer.name in layers_to_modify:

                        kernels = layer.get_weights()[0]
                        biases = layer.get_weights()[1]
                        kernels_extra_channel = np.concatenate((kernels,),
                                                               axis=-2)  # For channels_last

                        target_layer.set_weights([kernels_extra_channel, biases])

                    else:
                        target_layer.set_weights(layer.get_weights())




        return resnet50_template