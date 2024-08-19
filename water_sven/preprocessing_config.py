#####Config
###Paths
#Input files
# file_p = '/mnt/datadisk/data/Sentinel2/raw/'
file_p = '/home/myuser/data/Sentinel2/raw_data/'

# only used for differentiating between urban and rural areas
# labels_f = '/mnt/datadisk/data/Projects/water/inputs/water_labels.csv'
labels_f = '/home/myuser/prj/inputs/water_labels.csv'

#define which files to use: 'all' (use all files in folder), 'U' or 'R' (use only urban/rural files from labels_f)
#also defines folder name in preprocessed folder - other than 'U' or 'R' no restriction is implemented
restrict_files = ['multispectral']
# for testing purposes - set to False for full dataset and int for amount of files
restrict_files_to_amount = False
#where to save them
# preprocessed_p = '/mnt/datadisk2/preprocessed/'
preprocessed_p = '/home/myuser/preprocessed_data/'

###Load or overwrite temp steps
overwrite = True

###Options
fill_method = 'mean'
#'Z', 'Z local' or False
geotiff_normalization = ['Z']
#loads stds and means from file - if False, calculates them
load_means_stds_f = False
#Dimension to crop of geotiff (crops from middle) - 996 is max
height = 996
width = 996
###Values at which to clip
#use: 'outlier' in conjunction with std_multiplier for clipping at std_multiplier x standard deviation, can provide a
#tuple as well (min, max)
clipping_values = 'outlier'
#recommended 2.5?, 3
std_multiplier = 2.5

# Note: rasterio.open(f).read() starts counting at 1, not 0
#Channels to load use e.g. [4,3,2] for RGB
# Sentinel 2 (1-C) all channels: [B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12, QA10, QA20, QA60]
# pretrained model BigEarthNet: "B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A"] 
# resulting indices in the array: [2, 3, 4, 8, 5, 6, 7, 12, 13, 9]
# orion efficient bigearthnet: 
                # self._inputB02,
                # self._inputB03,
                # self._inputB04,
                # self._inputB05,
                # self._inputB06,
                # self._inputB07,
                # self._inputB08,
                # self._inputB8A,
                # self._inputB11,
                # self._inputB12,
# resulting indices in the array: [2, 3, 4, 5, 6, 7, 8, 9, 12, 13]
channel_l = [2, 3, 4, 5, 6, 7, 8, 9, 12, 13]

###Replaces NaN values
#Use a number, a 1D np.array, or 'local channel mean' (uses mean of the channel) (recommended).
replace_nan_value = 'local channel mean'
###Missing values: drop files and delete files over specified percentage of NaNs (only implemented in geotiff_no
drop_perc_NaNs = 5
delete_perc_NaNs = False
