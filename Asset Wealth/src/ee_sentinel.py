#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

sys.path.append("..")

import os
import functools
import time
from csv import DictReader
import json

import ee
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

from src.data_utils import truncate

from src.config import csv_path
from src.config import gdrive_dir_s2
from src.config import download_path_s2
from src.config import country_code_map

ee.Authenticate()
ee.Initialize()


def bounding_box(loc:ee.Geometry.Point, urban_rural:int, urban_radius:int, rural_radius:int):
    '''
    Function to get a square around point of interest
    Rural : 10 km Radius
    Urban : 2 km Radius
    Args:
        loc(ee.Geometry.Point): Geolocation of Cluster (from DHS Survey)
        urban_rural(int):       Binary Encoding for Region Type: 0 = urban, 1 = rural
        urban_radius(int):      Radius around Coordinates for Urban Regions in Meter
        rural_radius(int):      Radius around Coordinates for Rural Regions in Meter
    Returns:
        intermediate_box (ee.Geometry):     bounding box around cluster coordinates
                                            with a size of 10x10km for rural/ 2x2km for urban
    '''
    if urban_rural == 0 or urban_rural == '0':
        size = urban_radius
        ur = 'u'
    else:
        size = rural_radius
        ur = 'r'

    intermediate_buffer = loc.buffer(size)  # buffer radius, half your box width in m
    intermediate_box = intermediate_buffer.bounds()  # Draw a bounding box around the circle
    return (intermediate_box, ur)


#
def maskClouds(img:ee.Image, MAX_CLOUD_PROBABILITY:int):
    '''
    Masking of clouds
    Args:
        img(ee.Image):              Sentinel 2 Image retrieved from ee
        MAX_CLOUD_PROBABILITY(int): %

    Returns:
        ee.Image: CloudMasked EarthEngine Image
    '''
    clouds = ee.Image(img.get('cloud_mask')).select('probability')
    isNotCloud = clouds.lt(MAX_CLOUD_PROBABILITY)
    return img.updateMask(isNotCloud)


def get_image(cluster:object, urban_radius:int, rural_radius:int, country_code:str, MAX_CLOUD_PROBABILITY:int):
    '''
    Extract Information about Cluster to get Sentinel2 Image for corresponding Year and Coordinates.
    Args:
        cluster(DictReader object):    Information about the Cluster (Cluster number, Coordinates, Survey Name, etc.)
        survey_name(str):           Name of the Survey (COUNTRY_YEAR)
        urban_radius(int):          Radius around Coordinates for Urban Regions in Meter
        rural_radius(int):          Radius around Coordinates for Rural Regions in Meter
        country_code(str):          ISO Code for Survey Country (COUNTRY)
        MAX_CLOUD_PROBABILITY(int): %

    Returns:
        Requests Image from Earth Engine. Files are named by the following pattern:
            Latitude_Longitude-begin-end-country_r/u_sidelength
            Koordinaten: 4 Nachkommastellen
            Datumsformat: YYYYMMDD
            Land: Offizielle 3 Buchstaben Abkürzung (ISO)
            Rural und Urban: durch u bzw r
            Side length: Seitenlänge (Größe) der Kachel in km mit einer Nachkommastelle
    '''
    # Get images collections
    s2Sr = ee.ImageCollection('COPERNICUS/S2')
    s2Clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')

    # Get time span
    year = cluster["SURVEY_YEAR"]
    if int(year) < 2016:
        START_DATE = ee.Date('2015-06-01')
        END_DATE = ee.Date('2016-07-01')
        date_range = '20150601-20160701'
    else:
        START_DATE = ee.Date(str(int(year)) + '-01-01')
        END_DATE = ee.Date(str(int(year)) + '-12-31')
        date_range = str(year) + '0101-' + str(year) + '1231'

    # Point of interest (longitude, latidude)
    lat_float = float(cluster["LATNUM"])
    lon_float = float(cluster["LONGNUM"])
    loc = ee.Geometry.Point([lon_float, lat_float])
    # Region of interest
    region, ur = bounding_box(loc, cluster['URBAN_RURA'], urban_radius, rural_radius)
    cluster_no = int(round(float(cluster["DHSCLUST"]), 0))
    s2Sr = s2Sr.filterBounds(region).filterDate(START_DATE, END_DATE)
    s2Clouds = s2Clouds.filterBounds(region).filterDate(START_DATE, END_DATE)

    # Join S2 with cloud probability dataset to add cloud mask.
    s2SrWithCloudMask = ee.Join.saveFirst('cloud_mask').apply(
        primary=s2Sr,
        secondary=s2Clouds,
        condition=ee.Filter.equals(
            leftField='system:index', rightField='system:index')
    )

    maskCloudsWithProb = functools.partial(maskClouds, MAX_CLOUD_PROBABILITY=MAX_CLOUD_PROBABILITY)
    s2CloudMasked = ee.ImageCollection(s2SrWithCloudMask).map(maskCloudsWithProb).median()
    s2CloudMasked = s2CloudMasked.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10' \
                                             , 'B11', 'B12']).clip(region)


    if ur == 'u':
        filename = str(truncate(lat_float, 4)) + '_' + \
                   str(truncate(lon_float, 4)) + '_' + \
                   str(date_range) + '_' + \
                   str(country_code) + '_' + \
                   ur + '_' + \
                   str(float(urban_radius / 1000))
    else:
        filename = str(truncate(lat_float, 4)) + '_' + \
                   str(truncate(lon_float, 4)) + '_' + \
                   str(date_range) + '_' + \
                   str(country_code) + '_' + \
                   ur + '_' + \
                   str(float(rural_radius / 1000))
    print(filename)
    task = ee.batch.Export.image.toDrive(**{
        'image': s2CloudMasked,
        'description': filename,
        'folder': 'sentinel',
        'scale': 10})
    print('Create', filename)
    task.start()
    while task.active():  # request status of task
        print('Polling for task (id: {}).'.format(task.id))
        time.sleep(5)  # set sleep timer for 5 sec if task is still active
    return loc


def get_survey_images(file_dir:str, survey_name:str, urban_radius:int, rural_radius:int, MAX_CLOUD_PROBABILITY:int):
    '''
    Get Sentinel2 Image for each Cluster and download from GoogleDrive.
    Args:
        file_dir(str):              Path to DHS Survey CSV File
        survey_name(str):           Name of the Survey (COUNTRY_YEAR)
        urban_radius(int):          Radius around Coordinates for Urban Regions in Meter
        rural_radius(int):          Radius around Coordinates for Rural Regions in Meter
        MAX_CLOUD_PROBABILITY(int): %
    '''
    with open(file_dir, 'r') as read_obj:
        # pass the file object to DictReader() to get the DictReader object
        dict_reader = DictReader(read_obj)
        # get a list of dictionaries from dct_reader
        clusters = list(dict_reader)
        if clusters[0]['COUNTRY'].replace('_', ' ') == 'Democratic Republic of Congo':
            country_code = 'COD'
        elif clusters[0]['COUNTRY'].replace('_', ' ') == 'Cote d\'Ivoire':
            country_code = 'CIV'
        elif clusters[0]['COUNTRY'].replace('_', ' ') == 'Burkina Faso':
            country_code = 'BFA'
        elif clusters[0]['COUNTRY'].replace('_', ' ') == 'Sierra Leone':
            country_code = 'SLE'
        elif clusters[0]['COUNTRY'].replace('_', ' ') == 'Tanzania':
            country_code = 'TJK'

        else:
            country_code = country_code_map[clusters[0]['COUNTRY'].replace('_', ' ')]
        for cluster in clusters:
            loc = get_image(cluster, urban_radius, rural_radius, country_code, MAX_CLOUD_PROBABILITY)
            if float(cluster["DHSCLUST"]) % 50 == 0:
                download_local(os.path.join(img_dir, survey_name))
        download_local(os.path.join(img_dir, survey_name))


def download_local(survey_dir:str):
    '''
    Download Images from GoogleDrive Folder.
    Args:
        survey_dir(str): Output Directory for Download
    '''
    # folder which want to download from Drive
    folder_id = gdrive_dir_s2

    if survey_dir[-1] != '/':
        survey_dir = survey_dir + '/'

    if not os.path.exists(survey_dir):
        os.mkdir(survey_dir)

    file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(folder_id)}).GetList()
    for i, file1 in enumerate(sorted(file_list, key=lambda x: x['title']), start=1):
        print('Downloading {} from GDrive ({}/{})'.format(file1['title'], i, len(file_list)))
        title = file1['title']
        if not os.path.exists(survey_dir + title):
            file1.GetContentFile(survey_dir + title)
            file1.Delete()
        else:
            count = 1
            while os.path.exists(survey_dir + title):
                title = title.split('.')[0] + '_' + str(count) + '.tif'
                count += 1
            file1.GetContentFile(survey_dir + title)
            file1.Delete()


# Main functions for getting the sentinel images; here: only the directory for each survey is created
def sentinel_img_survey(img_dir:str, csv_dir:str, sentinel_done:str, urban_radius:int,
                        rural_radius:int, MAX_CLOUD_PROBABILITY:int):
    '''
    Iterate over Survey CSVs and get Sentinel2 Images for each Cluster.
    Args:
        img_dir(str):               Path to Directory where the Sentinel Images are stored
        csv_dir(str):               Path to Directory where DHS CSV Files are stored
        sentinel_done(str):         Filepath for File to document for which Surveys were are already completed
        urban_radius(int):          Radius around Coordinates for Urban Regions in Meter
        rural_radius(int):          Radius around Coordinates for Rural Regions in Meter
        MAX_CLOUD_PROBABILITY(int): %
    '''
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    if not os.path.isfile(sentinel_done):
        open(sentinel_done, 'a').close()

    csv_directory = os.listdir(csv_dir)
    img_directory = os.listdir(img_dir)

    for file in csv_directory:
        if file.endswith('.csv'):
            filename = file[:file.rfind('.')]
            print(filename)
            # Check if survey is already done we skip this survey (sentinel_done file has to be edited manually)
            with open(sentinel_done) as f:
                if not filename in f.read():
                    survey_name = file[:file.rfind('.')]
                    survey_dir = os.path.join(img_dir, survey_name)
                    if not os.path.exists(survey_dir):
                        os.makedirs(survey_dir)
                    file_dir = os.path.join(csv_dir, file)
                    get_survey_images(file_dir, survey_name, urban_radius, rural_radius, MAX_CLOUD_PROBABILITY)
                    # Add survey to txt file which stores all surveys which are done to avoid downloading them again if you reload the program
                    file1 = open(sentinel_done, "a")  # append mode
                    file1.write(file + "\n")
                    file1.close()
                    print(file, 'finished')


# Main Part

# Parameter
urban_radius = 2000  # meter
rural_radius = 10000  # meter
MAX_CLOUD_PROBABILITY = 20  # %

# Paths
# Path to Label Data
csv_dir = csv_path
# Directory where the sentinel images are stored
img_dir = download_path_s2
# Path to txt files which contains all surveys for which all images are already retrieved
sentinel_done =  "./sentinel_done.txt"

# Functions
gauth = GoogleAuth()
gauth.LoadCredentialsFile("mycreds.txt")
drive = GoogleDrive(gauth)

sentinel_img_survey(img_dir, csv_dir, sentinel_done, urban_radius, rural_radius, MAX_CLOUD_PROBABILITY)
