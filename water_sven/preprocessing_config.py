#####Config
###Paths
#Input files
file_p = '/mnt/datadisk2/satellite_data/sentinel2/raw/'
labels_f = '/mnt/datadisk/data/Projects/water/inputs/water_labels.csv'
#define which files to use: 'all' (use all files in folder), 'U' or 'R' (use only urban/rural files from labels_f)
restrict_files = ['U']
# file_p = '/mnt/datadisk/data/Sentinel2/raw/'
#where to save them
preprocessed_p = '/mnt/datadisk2/preprocessed/'
###Load or overwrite temp steps
overwrite = True
###Options
fill_method = 'mean'
#'Z', 'Z local' or False
geotiff_normalization = ['Z']
#Dimension to crop of geotiff (crops from middle) - 996 is max
height = 200
width = 200
###Values at which to clip
#use: 'outlier' in conjunction with std_multiplier for clipping at std_multiplier x standard deviation, can provide a
#tuple as well (min, max)
clipping_values = (0, 3000)
#recommended 2 or 3
std_multiplier = 2.5
#Channels to load use e.g. [4,3,2] for RGB
channel_l = [4, 3, 2]
###Replaces NaN values
#Use a number, a 1D np.array, or 'local channel mean' (uses mean of the channel) (recommended).
replace_nan_value = 'local channel mean'
###Missing values: drop files and delete files over specified percentage of NaNs (only implemented in geotiff_no
drop_perc_NaNs = 5
delete_perc_NaNs = 10
