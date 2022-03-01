import os

import numpy as np
from osgeo import gdal
from osgeo import osr

def pad_band_array(geo_tiff_pixels: np.array, band_pixels: np.array):
    '''

    Args:
        geo_tiff_pixels:    np.array
                            2D Numpy Array containing pixel values for one band
                            of urban geotiff
        band_pixels:        np.array
                            2D Numpy Array with rural shape (2004,2003),
                            filled with zeros

    Returns: band_pixels:   2D np.array with shape (2004,2003): contains urban array (401,401)
                            padded with zeros

    '''
    xsize = geo_tiff_pixels.shape[0]
    ysize = geo_tiff_pixels.shape[1]
    if xsize%2 == 1:
        xmin = (xsize + 1) / 2
        xmax = (xsize - 1) / 2
    else:
        xmin = xmax = xsize / 2
    if ysize%2 == 1:
        ymin = (ysize + 1) / 2
        ymax = (ysize - 1) / 2
    else:
        ymin = ymax = ysize / 2

    band_pixels[int(band_pixels.shape[0]/2-xmin):int(band_pixels.shape[0]/2+xmax),
    int(band_pixels.shape[1]/2-ymin):int(band_pixels.shape[1]/2+ymax)] = geo_tiff_pixels
    return(band_pixels)

def pad_urban_image(input_path:str, output_path:str, image_size=(2004,2004)):
    '''

    Args:
        input_path: Path to urban GeoTIFF to pad
        output_path: Output path for padded GeoTIFF
        image_size: Target size for output (usually (13,2004,2003)) to fit rural GeoTIFFs

    Returns:

    '''


    band_pixel_array = [np.zeros((image_size), dtype=np.uint8) for i in range(0, 13)]

    topad = gdal.Open(input_path)
    gt = topad.GetGeoTransform()
    Arraytopad = topad.ReadAsArray()

    width = topad.RasterXSize
    height = topad.RasterYSize
    minx = gt[0]
    miny = gt[3] + width * gt[4] + height * gt[5]
    maxx = gt[0] + width * gt[1] + height * gt[2]
    maxy = gt[3]

    nx = image_size[0]
    ny = image_size[1]
    xres = (maxx - minx) / float(nx)
    yres = (maxy - miny) / float(ny)
    geotransform = (minx, xres, 0, maxy, 0, -yres)
    if Arraytopad.shape[1] > 2004 or Arraytopad.shape[2] > 2004:
        Arraytopad = Arraytopad[:, :2004, :2004]
    for index, band in enumerate(band_pixel_array):
         band_pixel_array[index] = pad_band_array(Arraytopad[index], band)

    dst_ds = gdal.GetDriverByName('GTiff').Create(output_path, ny, nx, 13, gdal.GDT_Byte)
    dst_ds.SetGeoTransform(geotransform)  # specify coords
    srs = osr.SpatialReference()  # establish encoding
    srs.ImportFromEPSG(4326)  # WGS84 lat/long
    dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file

    for index, band in enumerate(band_pixel_array):
        dst_ds.GetRasterBand(index+1).WriteArray(band)

    dst_ds.FlushCache()  # write to disk
    dst_ds = None

def main(img_path:str, output_path:str):
    '''

    Args:
        img_path: Path to image data
        output_path: Path where resized images are to be stored

    Returns:

    '''
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    img_list = [img for img in os.listdir(img_path)if img.endswith('.tif') and img not in os.listdir(output_path)]
    for img in img_list:
        pad_urban_image(os.path.join(img_path,img),os.path.join(output_path, img))

if __name__ == '__main__':
    main(img_path = '/mnt/datadisk/data/Sentinel2/preprocessed/asset/urban_rural/train',
         output_path='/mnt/datadisk/data/Sentinel2/preprocessed/asset/urban_rural/train_padded'
         )

