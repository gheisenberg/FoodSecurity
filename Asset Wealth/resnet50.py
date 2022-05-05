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
from tensorflow.keras.applications.resnet_v2 import ResNet50V2

import numpy as np


class ResNet50v2_hyperspectral():
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

        self.hs_inputs = Input(shape=(self.img_h, self.img_w, self.channels), name='input')

    # adapted from https://github.com/keras-team/keras/blob/master/keras/applications/resnet.py to fit
    # hyperspectral input images
    def load_resnet50v2(self):
        '''
        Returns a Resnet50v2 `keras.Model` instance fitted to Hyperspectral/RGB image input
        '''

        def ResNet(stack_fn,
                   use_bias,
                   model_name='resnet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classifier_activation='softmax',
                   **kwargs):
            """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.
            Args:
            stack_fn: a function that returns output tensor for the
              stacked residual blocks.
            use_bias: whether to use biases for convolutional layers or not
              (True for ResNet and ResNetV2, False for ResNeXt).
            model_name: string, model name.
            input_tensor: optional Keras tensor
              (i.e. output of `layers.Input()`)
              to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
              if `include_top` is False (otherwise the input shape
              has to be `(224, 224, 3)` (with `channels_last` data format)
              or `(3, 224, 224)` (with `channels_first` data format).
              It should have exactly 3 inputs channels.
            pooling: optional pooling mode for feature extraction
              when `include_top` is `False`.
              - `None` means that the output of the model will be
                  the 4D tensor output of the
                  last convolutional layer.
              - `avg` means that global average pooling
                  will be applied to the output of the
                  last convolutional layer, and thus
                  the output of the model will be a 2D tensor.
              - `max` means that global max pooling will
                  be applied.
            classifier_activation: A `str` or callable. The activation function to use
              on the "top" layer. Ignored unless `include_top=True`. Set
              `classifier_activation=None` to return the logits of the "top" layer.
              When loading pretrained weights, `classifier_activation` can only
              be `None` or `"softmax"`.
            **kwargs: For backwards compatibility only.
            Returns:
            A `keras.Model` instance.
            """

            # Determine proper input shape
            img_input = Input(input_shape)

            bn_axis = 3

            x = ZeroPadding2D(
                padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
            x = Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

            x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
            x = MaxPooling2D(3, strides=2, name='pool1_pool')(x)

            x = stack_fn(x)

            if pooling == 'avg':
                x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
            elif pooling == 'max':
                x = layers.GlobalMaxPooling2D(name='max_pool')(x)

            x = layers.Flatten(name='flatten')(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dense(1)(x)

            # Create model.
            model = Model(img_input, x, name=model_name)

            return model

        def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
            """A residual block.
            Args:
            x: input tensor.
            filters: integer, filters of the bottleneck layer.
            kernel_size: default 3, kernel size of the bottleneck layer.
            stride: default 1, stride of the first layer.
            conv_shortcut: default True, use convolution shortcut if True,
                otherwise identity shortcut.
            name: string, block label.
            Returns:
            Output tensor for the residual block.
            """
            bn_axis = 3
            if conv_shortcut:
                shortcut = Conv2D(
                    4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
                shortcut = BatchNormalization(
                    axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
            else:
                shortcut = x
            x = BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name=name + '_preact_bn')(x)
            x = Activation('relu', name=name + '_preact_relu')(x)

            x = Conv2D(filters, 1, strides=stride, use_bias=False, name=name + '_1_conv')(x)
            x = BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
            x = Activation('relu', name=name + '_1_relu')(x)

            x = Conv2D(
                filters, kernel_size, padding='SAME', use_bias=False, name=name + '_2_conv')(x)
            x = BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
            x = Activation('relu', name=name + '_2_relu')(x)

            x = Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
            x = BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

            x = Add(name=name + '_add')([shortcut, x])
            x = Activation('relu', name=name + '_out')(x)
            return x

        def stack1(x, filters, blocks, stride1=2, name=None):
            """A set of stacked residual blocks.
            Args:
            x: input tensor.
            filters: integer, filters of the bottleneck layer in a block.
            blocks: integer, blocks in the stacked blocks.
            stride1: default 2, stride of the first layer in the first block.
            name: string, stack label.
            Returns:
            Output tensor for the stacked blocks.
            """
            x = block1(x, filters, stride=stride1, name=name + '_block1')
            for i in range(2, blocks + 1):
                x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
            return x

        def block2(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
            """A residual block.
            Args:
              x: input tensor.
              filters: integer, filters of the bottleneck layer.
              kernel_size: default 3, kernel size of the bottleneck layer.
              stride: default 1, stride of the first layer.
              conv_shortcut: default False, use convolution shortcut if True,
                otherwise identity shortcut.
              name: string, block label.
            Returns:
            Output tensor for the residual block.
            """
            bn_axis = 3

            preact = BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name=name + '_preact_bn')(x)
            preact = Activation('relu', name=name + '_preact_relu')(preact)

            if conv_shortcut:
                shortcut = Conv2D(
                    4 * filters, 1, strides=stride, name=name + '_0_conv')(preact)
            else:
                shortcut = MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

            x = Conv2D(
                filters, 1, strides=1, use_bias=False, name=name + '_1_conv')(preact)
            x = BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
            x = Activation('relu', name=name + '_1_relu')(x)

            x = ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
            x = Conv2D(
                filters,
                kernel_size,
                strides=stride,
                use_bias=False,
                name=name + '_2_conv')(x)
            x = BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
            x = Activation('relu', name=name + '_2_relu')(x)

            x = Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
            x = Add(name=name + '_out')([shortcut, x])
            return x

        def stack2(x, filters, blocks, stride1=2, name=None):
            """A set of stacked residual blocks.
            Args:
              x: input tensor.
              filters: integer, filters of the bottleneck layer in a block.
              blocks: integer, blocks in the stacked blocks.
              stride1: default 2, stride of the first layer in the first block.
              name: string, stack label.
            Returns:
              Output tensor for the stacked blocks.
            """
            x = block2(x, filters, conv_shortcut=True, name=name + '_block1')
            for i in range(2, blocks):
                x = block2(x, filters, name=name + '_block' + str(i))
                x = block2(x, filters, stride=stride1, name=name + '_block' + str(blocks))
            return x

        def block3(x,
                   filters,
                   kernel_size=3,
                   stride=1,
                   groups=32,
                   conv_shortcut=True,
                   name=None):
            """A residual block.
            Args:
            x: input tensor.
            filters: integer, filters of the bottleneck layer.
            kernel_size: default 3, kernel size of the bottleneck layer.
            stride: default 1, stride of the first layer.
            groups: default 32, group size for grouped convolution.
            conv_shortcut: default True, use convolution shortcut if True,
                otherwise identity shortcut.
            name: string, block label.
            Returns:
            Output tensor for the residual block.
            """
            bn_axis = 3

            if conv_shortcut:
                shortcut = Conv2D(
                    (64 // groups) * filters,
                    1,
                    strides=stride,
                    use_bias=False,
                    name=name + '_0_conv')(x)
                shortcut = BatchNormalization(
                    axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)

            else:
                shortcut = x

            x = Conv2D(filters, 1, use_bias=False, name=name + '_1_conv')(x)
            x = BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
            x = Activation('relu', name=name + '_1_relu')(x)

            c = filters // groups
            x = ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
            x = DepthwiseConv2D(
                kernel_size,
                strides=stride,
                depth_multiplier=c,
                use_bias=False,
                name=name + '_2_conv')(x)
            x_shape = backend.shape(x)[:-1]
            x = backend.reshape(x, backend.concatenate([x_shape, (groups, c, c)]))
            x = Lambda(
                lambda x: sum(x[:, :, :, :, i] for i in range(c)),
                name=name + '_2_reduce')(x)
            x = backend.reshape(x, backend.concatenate([x_shape, (filters,)]))
            x = BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
            x = Activation('relu', name=name + '_2_relu')(x)

            x = Conv2D(
                (64 // groups) * filters, 1, use_bias=False, name=name + '_3_conv')(x)
            x = BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

            x = Add(name=name + '_add')([shortcut, x])
            x = Activation('relu', name=name + '_out')(x)
            return x

        def stack3(x, filters, blocks, stride1=2, groups=32, name=None):
            """A set of stacked residual blocks.
            Args:
            x: input tensor.
            filters: integer, filters of the bottleneck layer in a block.
            blocks: integer, blocks in the stacked blocks.
            stride1: default 2, stride of the first layer in the first block.
            groups: default 32, group size for grouped convolution.
            name: string, stack label.
            Returns:
            Output tensor for the stacked blocks.
            """
            x = block3(x, filters, stride=stride1, groups=groups, name=name + '_block1')
            for i in range(2, blocks + 1):
                x = block3(
                    x,
                    filters,
                    groups=groups,
                    conv_shortcut=False,
                    name=name + '_block' + str(i))
            return x

        def ResNet50(input_tensor=None,
                     input_shape=None,
                     pooling=None,
                     **kwargs):
            """Instantiates the ResNet50 architecture."""

            def stack_fn(x):
                x = stack1(x, 64, 3, stride1=1, name='conv2')
                x = stack1(x, 128, 4, name='conv3')
                x = stack1(x, 256, 6, name='conv4')
                return stack1(x, 512, 3, name='conv5')

            return ResNet(stack_fn, True, 'resnet50', input_tensor, input_shape, pooling, **kwargs)

        resnet50v2_template = ResNet50(None, (self.img_w, self.img_h,self.channels), None)

        # Adjust weights to 13 input channels
        ## adapted from https://stackoverflow.com/questions/53251827/pretrained-tensorflow-model-rgb-rgby-channel-extension
        if self.channels > 3:
            layers_to_modify = ['block1_conv1']  # Turns out the only layer that changes
        # shape due to 4th to 13th channel is the first convolution layer.
        else:
            layers_to_modify = []
        for layer in self.pretrained_model.layers:  # pretrained Model and template have the same
            # layers, so it doesn't matter which to
            # iterate over.

            if layer.get_weights():  # Skip input, pooling and no weights layers

                target_layer = resnet50v2_template.get_layer(name=layer.name)
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
        resnet50v2_template.get_weights()

        return resnet50v2_template