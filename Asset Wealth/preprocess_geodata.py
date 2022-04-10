import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from osgeo import osr
from osgeo import gdal
import rasterio
from data_utils import create_splits, calc_mean, calc_std


def normalize_resize(img: str, img_path:str,  input_height:str, input_width:str, clipping_values: list, means: np.ndarray, stds: np.ndarray):
    '''
    Normalize an image based on the means and standard deviation of its split and resize it to the given input size.
    Args:
        img (str): Image name
        img_path (str): Path to image data
        input_height (int): pixel height of input
        input_width (int): pixel width of input
        clipping_values (list): interval of min and max values for clipping

    Returns:
        array(np.array): Normalized and resized Image Array
    '''
    ds = gdal.Open(os.path.join(img_path,img))
    geotransform = ds.GetGeoTransform()

    array = ds.ReadAsArray()
    
    array[np.isnan(array)] = 0
    assert not np.any(np.isnan(array)), "Float"

    # Clipping
    array = np.clip(array, a_min=clipping_values[0], a_max=clipping_values[1])

    assert not np.any(np.isnan(array)), "After clipping"

    # Resize the array to ensure that all image arrays have the same size
    array = np.resize(array, (array.shape[0], input_height, input_width))
    # Normalize the array
    array = ((array.transpose(1, 2, 0) - means) /stds).transpose(2, 0, 1)
    assert not np.any(np.isnan(array)), "Normalize"

    #return normalized and resized image array
    
    #Create new geotiff
    dst_ds = gdal.GetDriverByName('GTiff').Create(os.path.join(img_path, img), input_height, input_width, len(array), gdal.GDT_Byte)

    dst_ds.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference()            # establish encoding
    srs.ImportFromEPSG(4326)                # WGS84 lat/long
    dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    for i in range(len(array)):
        dst_ds.GetRasterBand(i+1).WriteArray(array[i])   # write band to the raster

    dst_ds.FlushCache()                     # write to disk
    dst_ds = None

def main(img_path:str,  input_height:str, input_width:str, clipping_values: list, channels:list):
    '''

    Args:
        img_path (str): Path to image data
        input_height (int): pixel height of input
        input_width (int): pixel width of input
        clipping_values (list): interval of min and max values for clipping
        channels (list): List of Channels to use. [] to use all channels.

    Returns:

    '''
    #Get Splits to Normalize Train/Val and Test separately
    X_train_val, X_test = train_test_split(os.listdir(img_path), test_size=0.20, random_state=42)
    
    # Calculate Means and Standard Deviations per Band
    train_val_means = calc_mean(img_path, X_train_val, input_height, input_width, clipping_values, channels)
    train_val_stds = calc_std(train_val_means, img_path, X_train_val, input_height, input_width, clipping_values, channels)
    
    test_means = calc_mean(img_path, X_test, input_height, input_width, clipping_values, channels)
    test_stds = calc_std(train_val_means, img_path, X_test, input_height, input_width, clipping_values, channels)
    
    # Normalize and Resize Images and save the preprocessed Images for Training
    for image in tqdm(X_train_val):
        normalize_resize(image, img_dir, input_height, input_width, clipping_values, train_val_means, train_val_stds)
    
    for image in tqdm(X_train_val):
        normalize_resize(image, img_dir, input_height, input_width, clipping_values, test_means, test_stds)
    

    
if __name__ == '__main__':
    main(
        img_path='/mnt/datadisk/data/Sentinel2/preprocessed/asset/rural/',  
        input_height=2000, 
        input_width=2000, 
        clipping_values=[0,3000], 
        channels=[]
    )
    main(
        img_path='/mnt/datadisk/data/Sentinel2/preprocessed/asset/urban_rural/',  
        input_height=1340, 
        input_width=1340, 
        clipping_values=[0,3000], 
        channels=[]
    )