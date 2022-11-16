import os
from tensorflow.keras import losses
from tensorflow.keras import metrics
########################################################################################################################
#                                   Performance Settings, Multi-GPU-usage
########################################################################################################################
# list of gpus to use
# (only use 1 GPU!) Otherwise I will kill your process!
# We all need to calculate on this machine - might get lowered to one, if there are bottlenecks!
gpus = [0, 1, 2]

########################################################################################################################
#                                           Verbosity settings
########################################################################################################################
#handles verbosity of the program (use 1 or above to get feedback!)
verbose = 1

########################################################################################################################
#                                           Base Folder
########################################################################################################################
### The folder where the project path and data is located, subfolder paths will be automatically generated
base_folder = '/mnt/datadisk/data/'
# The folder where your project files are located
prj_folder = '/mnt/datadisk/data/Projects/water/'
# train history will be saved in a subfolder of the project path (base_folder + /projects/water/)
# assign a name according to your group, to separate your results from all others! Create this folder manually!
trainHistory_subname = 'trainHistory_aug_cat/'
labels_f = os.path.join(prj_folder, 'water_labels_urban.csv')

########################################################################################################################
#                                            Run Name
########################################################################################################################
### The name of the model run gets generated by mutiple Settings (e.g. model_name, normalization and augmentation
# settings and many more, it will be created as a folder where all results model checkpoints, evaluation charts etc. pp
# are saved
# run_name_custom_string is a custom string you can provide to add to the run name
run_name_custom_string = ''


########################################################################################################################
#                                       Dataset parameters
########################################################################################################################
# 3 Main label categories - everything else is thrown away
# # label_name = ['source of drinking water (weights): max']
# label_name = ['source of drinking water (simplified): max', 'source of drinking water (categorized by type): max',
#                 'source of drinking water (categorized urban): max']
# label_name = ['source of drinking water (categorized): max', 'source of drinking water (weights): max',
#               'location of source for water (refurbished): max']
# label_name = ["source of drinking water (categorized piped): max", "source of drinking water (categorized piped 2): max",
#               "source of drinking water (categorized piped 3): max"]
label_name = ['distance categories cluster', 'distance categories cluster 2']#asd
# label_name = ['weighting (location)', 'time to get to water source + penalty',
#               'time to get to water source (refurbished) + penalty', 'time to get to water source + penalty (capped)']
# label_name = ['source of drinking water (weights)']
# [Minimum, maximum values] for clipping above and below those pixel values
clipping_values = [0, 3000]
# Channels (define channels which should be used, if all should be used provide an empty list = [])
channels = [4, 3, 2]
channel_size = len(channels)


########################################################################################################################
#                                  Basic neural network parameters
########################################################################################################################
### Maximum amount of epochs
epochs = 200
### Learning rate (to start with - might get dynamically lowered with callback options)
lr = 1e-4
# How many pictures are used to train before readjusting weights
batch_size = 64
### The model to use
# available are vgg16/19, resnet50/152, inceptionv3, xception, densnet121/201
model_name = 'vgg19'
# loss function
#keras losses and metrics imports - do not initialize!!! Need to be called in strategy scope
#CategoricalCrossentropy or MeanSquaredError
loss = losses.CategoricalCrossentropy
#CategoricalAccuracy or RootMeanSquaredError
metrics_l = [metrics.CategoricalAccuracy]
#categoricall or regression
type_m = 'categorical'
test_mode = False
#Z-normalization
label_normalization = False
#boxcox
label_transform = False
split = 'random split'
# chose your optimizer
optimizer = "SGD"
# momentum influences the lr (high lr when big changes occur and low lr when low changes occur)
momentum = 0.9

### The shape of X and Y with ((batch_size, height, width, channels), (batch_size, number of labels))
input_shape = (batch_size, 200, 200, channel_size)

### CNN settings which are parameters of the tf.keras.applications Model
# 'include_top': Use the same top layers (aka final layers with softmax on output classes) - should always be "False" here
# 'weights': 'imagenet' or False - transfer learning option - will be overwritten if model weights are given in
# load_model_weights. Only takes effect with include_top=False
# classifier activation: A str or callable. The activation function to use on the "top" layer. Ignored unless include_top=True.
# Set classifier_activation=None to return the logits of the "top" layer.
# When loading pretrained weights, classifier_activation can only be None or "softmax".
# others kinda explain themselves
cnn_settings_d = {'include_top': False, 'weights': 'imagenet', 'input_tensor': None,
                   'input_shape': (input_shape[1], input_shape[2], input_shape[3]), 'pooling': False}
                  #'classifier_activation': 'softmax'}
# Use dropout on top layers - use 0 to 1
dropout_top_layers = 0
# Make weights trainable. Unfreezes layers beginning at top layers. Use 0 to 100 (percent)
unfreeze_layers_perc = 100
### Use custom top layers (necessary when using transferlearning)
# 2 layers right now. Their neurons can be adjusted
add_custom_top_layers = True
# define how many hidden layers shall be added as top layers (len(neurons_l)) and how many neurons they should have (int)
neurons_l = [1024, 512]



########################################################################################################################
#                                       Callback options
########################################################################################################################
# automatically decrease lr (first value True) if there was no decrease in loss after x epochs (2nd value)
# 3rd multiplicator for lr
auto_adjust_lr = (False, 4, 0.8)
# model stops (first value True) when loss doesnt decrease over epochs (2nd value)
early_stopping = (True, 15)


########################################################################################################################
#                       ImageDataGenerator (IDG - Keras) Settings
########################################################################################################################
### Use 'ImageDataGenerator' or False - Shannons generator gets used when False
generator = 'ImageDataGenerator'

### Normalization settings for IDG
# only featurewise settings and zca_whitening get fitted or respectively (Mean, standarddeviation, PCA) get precalculated
# IDG.fit() only gets called for parameters in here
# dict with sub dicts to be able to use in a testing strategy
# these get applied to train, test and validation data
IDG_normalization_d = {
   # 'samplewise': {'samplewise_center': True,  # Boolean. Set each sample mean to 0. #needs IDG.fit()
   #                'samplewise_std_normalization': True},  # needs IDG.fit()
    'featurewise': {'featurewise_center': True, #Boolean. Set input mean to 0 over the dataset, feature-wise. #needs IDG.fit()
    'featurewise_std_normalization': True},  # needs IDG.fit()
    # 'manually_normalize': {},  # dummy for no normalization
    # 'rescale': 100,
    # 'zca_whitening': True  # Boolean. Apply ZCA whitening. #needs IDG.fit(), takes ages and kills process regularly
}
### ZCA is extremly expensive - thus we only wanna calculate it on a subset of input images
# use a number from 1 to 100 (percent)
# Experimental state - usually crashes!!!
zca_whitening_perc_fit = 1

### Image Augmentation settings
# all possible values below
# accuracy decreases with to many Augmentation settings, though - why?
# dict of dicts to be able to test multiple settings in a testing strategy
# augmentation only gets applied to train data
IDG_augmentation_settings_d = {'subset1': {
        #'brightness_range': [0.9, 1.1], #Tuple or list of two floats. Range for picking a brightness shift value from.
        #'shear_range': 0.2, #Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
        #'zoom_range': 0.2,
        #'channel_shift_range': 0.3,
        'horizontal_flip': True,
        #'vertical_flip': True,
        #'rotation_range': 20, #Int. Degree range for random rotations.
        #'width_shift_range': 0.2,
        #'height_shift_range': 0.2
        },

'subset13': {
        #'brightness_range': [0.9, 1.1], #Tuple or list of two floats. Range for picking a brightness shift value from.
        'shear_range': 0.2, #Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
        'zoom_range': 0.2,
        #'channel_shift_range': 0.3,
        'horizontal_flip': True,
        #'vertical_flip': True,
        # 'rotation_range': 20, #Int. Degree range for random rotations.
        # 'width_shift_range': 0.2,
        # 'height_shift_range': 0.2
    },
    'subset2': {
        # 'brightness_range': [0.9, 1.1], #Tuple or list of two floats. Range for picking a brightness shift value from.
        # 'shear_range': 0.2, #Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
        # 'zoom_range': 0.2,
        # 'channel_shift_range': 0.3,
        'horizontal_flip': True,
        'vertical_flip': True,
        # 'rotation_range': 20, #Int. Degree range for random rotations.
        # 'width_shift_range': 0.2,
        # 'height_shift_range': 0.2
    },
'subset9': {
        #'brightness_range': [0.9, 1.1], #Tuple or list of two floats. Range for picking a brightness shift value from.
        'shear_range': 0.2, #Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
        'zoom_range': 0.2,
        'channel_shift_range': 0.3,
        'horizontal_flip': True,
        #'vertical_flip': True,
        # 'rotation_range': 20, #Int. Degree range for random rotations.
        # 'width_shift_range': 0.2,
        # 'height_shift_range': 0.2
    },
'subset11': {
        #'brightness_range': [0.9, 1.1], #Tuple or list of two floats. Range for picking a brightness shift value from.
        #'shear_range': 0.2, #Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
        'zoom_range': 0.2,
        'channel_shift_range': 0.3,
        'horizontal_flip': True,
        #'vertical_flip': True,
        # 'rotation_range': 20, #Int. Degree range for random rotations.
        # 'width_shift_range': 0.2,
        # 'height_shift_range': 0.2
    },
'subset12': {
        #'brightness_range': [0.9, 1.1], #Tuple or list of two floats. Range for picking a brightness shift value from.
        'shear_range': 0.2, #Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
        #'zoom_range': 0.2,
        'channel_shift_range': 0.3,
        'horizontal_flip': True,
        #'vertical_flip': True,
        # 'rotation_range': 20, #Int. Degree range for random rotations.
        # 'width_shift_range': 0.2,
        # 'height_shift_range': 0.2
    },
'subset14': {
        #'brightness_range': [0.9, 1.1], #Tuple or list of two floats. Range for picking a brightness shift value from.
        #'shear_range': 0.2, #Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
        'zoom_range': 0.2,
        #'channel_shift_range': 0.3,
        'horizontal_flip': True,
        #'vertical_flip': True,
        # 'rotation_range': 20, #Int. Degree range for random rotations.
        # 'width_shift_range': 0.2,
        # 'height_shift_range': 0.2
    },
'subset15': {
        #'brightness_range': [0.9, 1.1], #Tuple or list of two floats. Range for picking a brightness shift value from.
        #'shear_range': 0.2, #Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
        #'zoom_range': 0.2,
        'channel_shift_range': 0.3,
        'horizontal_flip': True,
        #'vertical_flip': True,
        # 'rotation_range': 20, #Int. Degree range for random rotations.
        # 'width_shift_range': 0.2,
        # 'height_shift_range': 0.2
    },
}
# #
# IDG_augmentation_settings_d = {'subset1': {
#         #'brightness_range': [0.9, 1.1], #Tuple or list of two floats. Range for picking a brightness shift value from.
#         #'shear_range': 0.2, #Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
#         #'zoom_range': 0.2,
#         #'channel_shift_range': 0.3,
#         'horizontal_flip': True,
#         #'vertical_flip': True,
#         #'rotation_range': 20, #Int. Degree range for random rotations.
#         #'width_shift_range': 0.2,
#         #'height_shift_range': 0.2
#         }}
########################################################################################################################
#                               Continuing from earlier runs
########################################################################################################################
### False or modelcheckp(oint) folder from which to load weights
load_model_weights = False
    # os.path.join(base_folder,
    #  "/mnt/datadisk/data/Projects/water/trainHistory_aug//source_of_drinking_water_(categorized_by_type)__max/vgg19_wimagenet_unfl100_d0_lr0.0001_momentum0.9_optimizerSGD_shear_0.2zoom_0.2channel_shift_0.3horizontal_flip_featurewise_1/modelcheckp/")


########################################################################################################################
#                               Evaluation Settings & Images
########################################################################################################################
### You can reload the best model epoch (True/False) - in that case evaluation is done on both, the best model epoch and
# the last one
# overrides other weights imports (e.g. imagenet in Keras models)
reload_best_weights_for_eval = True
### You can show and/or save your augmented images to become an idea of what actually goes into the model
# False or Number of images (for train, val and test gen)
save_augmented_images = 15
tensorboard = True