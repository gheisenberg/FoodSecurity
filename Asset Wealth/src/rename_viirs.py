import sys

sys.path.append("../../../Asset Wealth")

import os
from osgeo import gdal
import rasterio

import pandas as pd
import numpy as np
from affine import Affine
from pyproj import Proj, transform
import json

from src.data_utils import truncate
from src.data_utils import combine_wealth_dfs

from src.config import download_path_viirs
from src.config import csv_path

def get_center_coords(img_path=str):
    '''Get center coordinates of a GeoTIFF.
    
    Args:
        img_path (str): Path to GeoTIFF

    Returns:
        long (float): Center Longitude Value of Image
        lat (float): Center Latitude Value of Image

    '''
    fname = img_path

    # Read raster
    with rasterio.open(fname) as r:
        T0 = r.transform  # upper-left pixel corner affine transform
        p1 = Proj(r.crs)
        A = r.read()  # pixel values

    # All rows and columns
    cols, rows = np.meshgrid(np.arange(A.shape[2]), np.arange(A.shape[1]))

    # Get affine transform for pixel centres
    T1 = T0 * Affine.translation(0.5, 0.5)
    # Function to convert pixel row/column index (from 0) to easting/northing at centre
    rc2en = lambda r, c: (c, r) * T1

    # All eastings and northings (there is probably a faster way to do this)
    eastings, northings = np.vectorize(rc2en, otypes=[float, float])(rows, cols)

    # Project all longitudes, latitudes
    p2 = Proj(proj='latlong', datum='WGS84')
    longs, lats = transform(p1, p2, eastings, northings)

    long = longs[int(longs.shape[0] / 2), 0]
    lat = lats[0, int(longs.shape[1] / 2)]
    return long, lat


def main(img_dir: str):
    '''Rename all VIIRS GeoTIFFs from DIS22 according to filename pattern:
    Latitude_Longitude_begin-end_COUNTRY_r/u_sidelength
    coordinates: 4 Nachkommastellen
    date format: YYYYMMDD
    country: Official 3 letters acronym (ISO)
    Rural/Urban: u or r
    side length: Sidelength (size) of tile in km with one decimal place.
    
    Args:
        img_dir: Path to image data
    '''
    img_list = [img for img in os.listdir(img_dir) if img.endswith('.tif')]
    wealth_df = combine_wealth_dfs(csv_path)
    not_found = []
    for img in img_list:

        cc = img[:2]
        year = img[2:6]
        lat_float, lon_float = get_center_coords(os.path.join(img_dir, img))

        country_year = wealth_df[(wealth_df.DHSCC == cc) & (wealth_df.DHSYEAR == float(year))]

        country_year['LATNUM'] = country_year.LATNUM.astype('str')
        country_year['LONGNUM'] = country_year.LONGNUM.astype('str')
        n = 3
        cluster = country_year[(country_year.LATNUM.str.startswith(str(lat_float)[:n])) | (
            country_year.LONGNUM.str.startswith(str(lon_float)[:n]))]
        if cluster.empty:
            print('Cluster not found')
            not_found.append(img)
            continue
        if len(cluster) > 1:
            n += 1
            cluster = country_year[(country_year.LATNUM.str.startswith(str(lat_float)[:n])) | (
                country_year.LONGNUM.str.startswith(str(lon_float)[:n]))]
        with open('./country_code_mapping.json', 'r') as fp:
            country_code_map = json.load(fp)
        if cluster.COUNTRY.iloc[0] == 'Democratic Republic of Congo':
            country_code = 'COD'
        elif cluster.COUNTRY.iloc[0] == 'Cote d\'Ivoire':
            country_code = 'CIV'
        elif cluster.COUNTRY.iloc[0] == 'Burkina Faso':
            country_code = 'BFA'
        elif cluster.COUNTRY.iloc[0] == 'Sierra Leone':
            country_code = 'SLE'
        elif cluster.COUNTRY.iloc[0] == 'Tanzania':
            country_code = 'TJK'
        else:
            country_code = country_code_map[cluster.COUNTRY.iloc[0]]
        lat = cluster.LATNUM.iloc[0]
        long = cluster.LONGNUM.iloc[0]
        if int(year) < 2016:
            date_range = '20150601-20160701'
        else:
            date_range = str(year) + '0101-' + str(year) + '1231'
        urban_rural = cluster.URBAN_RURA.iloc[0]
        if urban_rural == 0 or urban_rural == '0':
            ur = 'u'
        else:
            ur = 'r'
        if ur == 'u':
            filename = str(truncate(lat_float, 4)) + '_' + \
                       str(truncate(lon_float, 4)) + '_' + \
                       str(date_range) + '_' + \
                       str(country_code) + '_' + \
                       ur + '_' + \
                       str(float(2))

        else:
            filename = str(truncate(lat_float, 4)) + '_' + \
                       str(truncate(lon_float, 4)) + '_' + \
                       str(date_range) + '_' + \
                       str(country_code) + '_' + \
                       ur + '_' + \
                       str(float(10))
        print(filename)
        dest_dir = os.path.join(img_dir, cluster.COUNTRY.iloc[0] + year)
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        os.rename(os.path.join(img_dir, img), os.path.join(dest_dir, filename + '.tif'))


if __name__ == '__main__':
    main(img_dir=download_path_viirs)