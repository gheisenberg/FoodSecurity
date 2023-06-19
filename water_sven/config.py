import os
from tensorflow.keras import losses
from tensorflow.keras import metrics
########################################################################################################################
#                                   Performance Settings, Multi-GPU-usage
########################################################################################################################
# list of gpus to use
# (only use 1 GPU!) Otherwise I will kill your process!
# We all need to calculate on this machine - might get lowered to one, if there are bottlenecks!
gpus = [0,1,2]


########################################################################################################################
#                                           Verbosity settings
########################################################################################################################
#handles verbosity of thzooe program (use 1 or above to get feedback!)
verbose = False
# 'debug' for debugging, 'info' for a bit verbosity, 'warning' for less verbosity
logging = 'info'


########################################################################################################################
#                                           Base Folder
########################################################################################################################
### The folder where the project path and data is located, subfolder paths will be automatically generated
#Note: Pathes need to have the structure of /path/to/file and a folder need to end with an '/'
base_folder = '/mnt/datadisk/data/'
# The folder where your project files are located
prj_folder = '/mnt/datadisk/data/Projects/water/'
# train history will be saved in a subfolder of the project path (base_folder + /projects/water/)
# assign a name according to your group, to separate your results from all others! Create this folder manually!
trainHistory_subname = 'trainH_XVU2/'
labels_f = prj_folder + '/inputs/water_labels_strat_split3.csv'
# img_path = '/mnt/datadisk/data/Sentinel2/preprocessed/water_new/'
# img_path = ['/mnt/datadisk2/preprocessed/R/996x996_c432_fillmean_m2.5_rlocal channel mean_clipvoutlier_normZ_f18977/',]
img_path = '/mnt/datadisk2/preprocessed/all/996x996_c432_fillmean_m2.5_rlocal channel mean_clipvoutlier_normZ_f31213/'
#use directory with multiple folders
load_subfolders = False
tmp_p = '/mnt/datadisk2/tmp/'


########################################################################################################################
#                                       Testing routines
########################################################################################################################
#define the split_col to use
splits_l = [
    # 'split: random 10splits urban 2012plus',
    # 'split: random 10splits urban 2012plus excluded drop surveys (ZAuEG)',
    # 'split: random 10splits urban 2012plus excluded outlier surveys',
    # 'split: out of country year urban 2012plus MZ in train',
    # 'split: out of country year urban 2012plus excluded drop surveys (ZAuEG) MZ in train',
    # 'split: out of country year urban 2012plus excluded outlier surveys MZ in train',
    # 'split: out of adm1 year all 2012plus',
    'split: random urban 2012plus excluded drop surveys (ZAuEG)',
    'split: non overlapping year urban 2012plus excluded drop surveys (ZAuEG)',
    # 'split: random urban 2012plus excluded drop surveys (ZAuEG)',
    # 'split: out of country all 2012plus MZ in train',
    # 'split: out of country year all MZ in train',
    # 'split: out of country year all excluded outlier surveys MZ in train',
    # 'split: out of country all MZ in test',
    # 'split: out of country all excluded drop surveys (ZAuEG) MZ in test',
    # 'split: out of country all excluded outlier surveys MZ in test',
    # 'split: out of adm1 year all',
    # 'split: out of adm1 year all excluded drop surveys (ZAuEG)',
    # 'split: out of adm1 year all excluded outlier surveys',
    # 'split: out of adm2 year all random',
    # 'split: out of adm2 year all random excluded drop surveys (ZAuEG)',
    # 'split: out of adm2 year all random excluded outlier surveys random',
    # 'split: random all',
    # 'split: random all excluded drop surveys (ZAuEG)',
    # 'split: random all excluded outlier surveys random',
]

# define the label column names with an appended number and specify a normalization, upper and lower std multiples to drop outliers
# (refer to load_labels() in water_w_regression.py for more information)
label_d = [
    'PCA w_location_weighting urban2', {'label normalization': 'Z',}
    # 'PCA w_weighting urban3': {'label normalization': 'Z',
    # 'max std': 2.5, 'drop min max value': False,
    # 'min std': -2.5},
    # 'label transform': 'boxcox',
                            ]

### Image Augmentation settings
# (IDG is slowing down like crazy and multiple augmentations for the same image create
# heavy distortions: only zoom and horizontal flip seem to work well)
# in tensorflow 2.8, tensorflow 2.9+ not working right now
# accuracy decreases with to many Augmentation settings, though - why?
# augmentation only gets applied to train data
IDG_augmentation_settings_d = {}
# {
#     # 'zoom_range': 0.2,
#     # 'horizontal_flip': True,
# }

# define the dimensions (img height and width) to use
dim_l = False #[50, 100, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 900, False]

###Combine all these testing routines in a list of lists (see above for detailed description)
#[[split_col, label_d, img_path, IDG_augmentation_settings_d, dim_l],]
# splitn_labeld_imgp_augmentation_dimension_l = [
#     ['split: out of country all (w TIF)', ['PCA w_location_weighting all2', {'label normalization': 'Z',}], img_path, IDG_augmentation_settings_d, dim_l],
# ]
# can also be created like this (e.g.; be careful though!):
# splitn_labeld_imgp_augmentation_dimension_l = \
#     list(zip([splits_l[0] for dim in dim_l], [label_d for dim in dim_l], [img_path for dim in dim_l],
#          [IDG_augmentation_settings_d for dim in dim_l], dim_l))
splitn_labeld_imgp_augmentation_dimension_l = \
    list(zip(splits_l, [label_d for spl in splits_l], [img_path for spl in splits_l],
         [IDG_augmentation_settings_d for spl in splits_l], [dim_l for spl in splits_l]))

########################################################################################################################
#                                       Further dataset parameters
########################################################################################################################
#removes all label data before 2012
label_min_year = 2012
# Channels use False for preprocessed images (define channels which should be used, if you want to use a subset of these channels)
channels = False


########################################################################################################################
#                                  Basic neural network parameters
########################################################################################################################
### Maximum amount of epochs
epochs = 100
### Learning rate (to start with - might get dynamically lowered with callback options)
lr = 0.0001
# How many pictures are used to train before readjusting weights
batch_size = 16
### The model to use
# available are vgg16/19, resnet50/152, inceptionv3, xception, densnet121/201
model_name = 'vgg19'
# loss function
#keras losses and metrics imports - do not initialize!!! Need to be called in strategy scope
#CategoricalCrossentropy or MeanSquaredError
loss = losses.MeanSquaredError
#CategoricalAccuracy or RootMeanSquaredError
metrics_l = [metrics.RootMeanSquaredError]
#categorical or regression
type_m = 'regression'
# chose your optimizer
optimizer = "SGD"
# momentum influences the lr (high lr when big changes occur and low lr when low changes occur)
momentum = 0.9


########################################################################################################################
#                                       CNN Settings
########################################################################################################################
### CNN settings which are parameters of the tf.keras.applications Model
# 'include_top': Use the same top layers (aka final layers with softmax on output classes) - should always be "False" here
# 'weights': 'imagenet' or False - transfer learning option - will be overwritten if model weights are given in
# load_model_weights. Only takes effect with include_top=False
# classifier activation: A str or callable. The activation function to use on the "top" layer. Ignored unless include_top=True.
# Set classifier_activation=None to return the logits of the "top" layer.
# When loading pretrained weights, classifier_activation can only be None or "softmax".
# others kinda explain themselves
cnn_settings_d = {'include_top': False, 'weights': 'imagenet', 'input_tensor': None, 'pooling': False}
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
auto_adjust_lr = (False, 1, 0.8)
# model stops (first value True) when loss doesnt decrease over epochs (2nd value)
early_stopping = (True, 8)


#use an int to limit data on that amount or False to use full dataset (and not testing mode)
test_mode = False
#only uses the first split for validation (way faster - for hyperparameter optimization and co)
dont_use_crossval = False
#set to True if denormalized (etc.) data should be reported
report_original_data = False


########################################################################################################################
#                               Continuing from earlier runs
########################################################################################################################
### False or modelcheckp(oint) folder from which to load weights
# Be very careful here! This should not be done between different models or different data splits! Information might
# leak from one to another! Not tested right now!
load_model_weights = False
# \
#     os.path.join(prj_folder, 'trainHistory_rural', 'PCA_w_location_weighting_rural1',
# '996x996_c432_fillmean_m2.5_rlocal_channel_mean_clipvoutlier_normZ_f18977vgg19_wimagenet_unfl100_d0_lr0.0001_momentum0.9_optimizerSGD__m_vgg19_17',
#                  'modelcheckp', '')


########################################################################################################################
#                               Evaluation Settings & Images
########################################################################################################################
### You can reload the best model epoch (True/False) - in that case evaluation is done on both, the best model epoch and
# the last one
# overrides other weights imports (e.g. imagenet in Keras models)
reload_best_weights_for_eval = True
### You can show and/or save your augmented images to become an idea of what actually goes into the model
# False or Number of images (for every split)
save_augmented_images = 5
#write tensorboard logs
tensorboard = True
