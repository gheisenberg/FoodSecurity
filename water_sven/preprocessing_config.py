#####Config
###Paths
#Input files
# file_p = '/mnt/datadisk2/satellite_data/sentinel2/raw/'
file_p = '/mnt/datadisk/data/Sentinel2/raw/'

# only used for differentiating between urban and rural areas
labels_f = '/mnt/datadisk/data/Projects/water/inputs/water_labels.csv'
#define which files to use: 'all' (use all files in folder), 'U' or 'R' (use only urban/rural files from labels_f)
#also defines folder name in preprocessed folder
restrict_files = ['all_new']
#where to save them
preprocessed_p = '/mnt/datadisk2/preprocessed/'
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
#Channels to load use e.g. [4,3,2] for RGB
channel_l = [4, 3, 2]
###Replaces NaN values
#Use a number, a 1D np.array, or 'local channel mean' (uses mean of the channel) (recommended).
replace_nan_value = 'local channel mean'
###Missing values: drop files and delete files over specified percentage of NaNs (only implemented in geotiff_no
drop_perc_NaNs = 5
delete_perc_NaNs = 10
