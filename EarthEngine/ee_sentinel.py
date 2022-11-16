#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

#sys.path.append("../../../Asset Wealth")

import os
import functools
import time
from csv import DictReader
import json
import multiprocessing as mp

import ee

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pydrive.files import ApiRequestError

# from data_utils import truncate

# from config import csv_path
# from config import gdrive_dir_s2
# from config import download_path_s2
# from config import country_code_map



###Quick Guide
#Initialize Drive to download files
#https://pythonhosted.org/PyDrive/quickstart.html
#start script on remote machine
#label csv at least needs: 'TIF_name' (the name of the file, otherwise generate a name), 'LATNUM', 'LONGNUM',
#'URBAN_RURA'
#apply_async may start thousands of tasks - that's why it's programmed to wait after 'parallel_dls' for the last download
#to end and for 60 mins after the last download has ended, because earlier downloads may take longer!
#Also, when using this apply_async you may create thousands of tasks for the google earth engine which are still running
#after the program terminated. (Not sure why pool.join does not wait for all workers to finish...).
#you may want to manually delete all tasks on the remote machine with 'earthengine task cancel all'
#downloaded!

###Options###
##Paths
label_csv = '/mnt/datadisk/data/Projects/water/water_labels.csv'
img_p = '/mnt/datadisk/data/Sentinel2/raw/'
##Parameter
#these should be identical and probably 5000 (yields 10x10km tiles)
urban_radius = 5000  # meter
rural_radius = 5000  # meter
#20 is suggested
MAX_CLOUD_PROBABILITY = 20  # %
#the minimum year of the questionnaire to download files
min_year = 2012
#Google stuff
gdrive_dir_s2 = '1P4FpvICI0S9vRs8mNvHxzqooK7zPGGnP'
parallel_dls = 150
#wait for 6 hours to complete all tasks and download them
final_waiting_time = 6
# Paths
# Path to Label Data
# csv_dir = csv_path
# Directory where the sentinel images are stored
# Path to txt files which contains all surveys for which all images are already retrieved
# sentinel_done = base_p + "sentinel_done.txt"


def bounding_box(loc:ee.Geometry.Point, urban_rural:int, urban_radius:int, rural_radius:int):
    '''Function to get a square around point of interest.
    Rural : 10 km Radius
    Urban : 2 km Radius
    
    Args:
        loc(ee.Geometry.Point): Geolocation of cluster (from DHS survey)
        urban_rural(int):       Binary encoding for type of region: 0 = urban, 1 = rural
        urban_radius(int):      Radius around coordinates for Urban regions in meter
        rural_radius(int):      Radius around coordinates for Rural regions in meter
    Returns:
        intermediate_box (ee.Geometry):     bounding box around cluster coordinates
                                            with a size of 10x10km for rural/ 2x2km for Urban
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
    '''Masking of clouds.
    
    Args:
        img(ee.Image):              Sentinel-2 image retrieved from ee
        MAX_CLOUD_PROBABILITY(int): %

    Returns:
        ee.Image: CloudMasked GoogleEarthEngine image
    '''
    clouds = ee.Image(img.get('cloud_mask')).select('probability')
    isNotCloud = clouds.lt(MAX_CLOUD_PROBABILITY)
    return img.updateMask(isNotCloud)


def get_image(cluster:object, urban_radius:int, rural_radius:int, country_code:str, MAX_CLOUD_PROBABILITY:int, drive):
    '''Extract Information about cluster to get Sentinel-2 image for corresponding year and coordinates.
    
    Args:
        cluster(DictReader object):    Information about the Cluster (cluster number, coordinates, survey name, etc.)
        survey_name(str):           Name of the survey (COUNTRY_YEAR)
        urban_radius(int):          Radius around coordinates for Urban regions in meter
        rural_radius(int):          Radius around coordinates for Rural regions in meter
        country_code(str):          ISO code for survey country (COUNTRY)
        MAX_CLOUD_PROBABILITY(int): %

    Returns:
        Requests Image from Earth Engine. Files are named by the following pattern:
            Latitude_Longitude_begin-end_COUNTRY_r/u_sidelength
            coordinates: 4 Nachkommastellen
            date format: YYYYMMDD
            country: Official 3 letters acronym (ISO)
            Rural/Urban: u or r
            side length: Sidelength (size) of tile in km with one decimal place.
    '''
    # Get images collections
    s2Sr = ee.ImageCollection('COPERNICUS/S2')
    s2Clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')

    # Get time span
    year = int(float(cluster["DHSYEAR"]))
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
    # cluster_no = int(round(float(cluster["DHSCLUST"]), 0))
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

    filename = cluster['TIF_name'][:-4]
    # if ur == 'u':
    #     filename = str(truncate(lat_float, 4)) + '_' + \
    #                str(truncate(lon_float, 4)) + '_' + \
    #                str(date_range) + '_' + \
    #                str(country_code) + '_' + \
    #                ur + '_' + \
    #                str(float(urban_radius / 1000))
    # else:
    #     filename = str(truncate(lat_float, 4)) + '_' + \
    #                str(truncate(lon_float, 4)) + '_' + \
    #                str(date_range) + '_' + \
    #                str(country_code) + '_' + \
    #                ur + '_' + \
    #                str(float(rural_radius / 1000))
    task = ee.batch.Export.image.toDrive(**{
        'image': s2CloudMasked,
        'description': filename,
        'folder': 'sentinel',
        'scale': 10})
    # print('Create', filename)
    t1 = time.time()
    task.start()
    while task.active():  # request status of task
        # print('Polling for task (id: {}).'.format(task.id))
        status = task.status()
        print(status['description'], status['state'], (time.time() - t1)/60, 'mins')
        time.sleep(60*5)  # set sleep timer for 5 sec if task is still active
    # print('created', filename, (time.time() - t1)/60)
    print('\n --------\ncreated', status['description'], status['id'], status['state'], (time.time() - t1)/60, 'mins')
    download_local(img_p, drive)
    return loc


def get_survey_images(file_dir:str, survey_name:str, urban_radius:int, rural_radius:int, MAX_CLOUD_PROBABILITY:int):
    '''Get Sentinel-2 image for each Cluster and download from GoogleDrive.
    
    Args:
        file_dir(str):              Path to DHS survey csv file
        survey_name(str):           Name of the survey (COUNTRY_YEAR)
        urban_radius(int):          Radius around coordinates for Urban regions in meter
        rural_radius(int):          Radius around coordinates for Rural regions in meter
        MAX_CLOUD_PROBABILITY(int): %
    '''
    with open(file_dir, 'r') as read_obj:
        # pass the file object to DictReader() to get the DictReader object
        dict_reader = DictReader(read_obj)
        # get a list of dictionaries from dct_reader
        clusters = list(dict_reader)
        # if clusters[0]['COUNTRY'].replace('_', ' ') == 'Democratic Republic of Congo':
        #     country_code = 'COD'
        # elif clusters[0]['COUNTRY'].replace('_', ' ') == 'Cote d\'Ivoire':
        #     country_code = 'CIV'
        # elif clusters[0]['COUNTRY'].replace('_', ' ') == 'Burkina Faso':
        #     country_code = 'BFA'
        # elif clusters[0]['COUNTRY'].replace('_', ' ') == 'Sierra Leone':
        #     country_code = 'SLE'
        # elif clusters[0]['COUNTRY'].replace('_', ' ') == 'Tanzania':
        #     country_code = 'TJK'
        #
        # else:
        #     country_code = country_code_map[clusters[0]['COUNTRY'].replace('_', ' ')]
        country_code = None
        for cluster in clusters:
            loc = get_image(cluster, urban_radius, rural_radius, country_code, MAX_CLOUD_PROBABILITY)
            #download files to local machine
            if float(cluster["DHSCLUST"]) % 2 == 0:
                download_local(os.path.join(img_p, survey_name))
        download_local(os.path.join(img_p, survey_name))


def download_local(survey_dir:str, drive):
    '''Download images from GoogleDrive folder.
    
    Args:
        survey_dir(str): Output directory for download
    '''
    # folder which want to download from Drive
    folder_id = gdrive_dir_s2

    if survey_dir[-1] != '/':
        survey_dir = survey_dir + '/'

    if not os.path.exists(survey_dir):
        os.mkdir(survey_dir)
    try:
        file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(folder_id)}).GetList()
        for i, file1 in enumerate(sorted(file_list, key=lambda x: x['title']), start=1):
            print('     Downloading {} from GDrive ({}/{})'.format(file1['title'], i, len(file_list)))
            title = file1['title']
            print('     dl', survey_dir + title)
            if not os.path.exists(survey_dir + title):
                file1.GetContentFile(survey_dir + title)
                file1.Delete()
            else:
                print('     This should not happen!')
                count = 1
                title2 = title
                while os.path.exists(survey_dir + title2):
                    title2 = title[:-4] + '_' + str(count) + '.tif'
                    count += 1
                print('     dl name', survey_dir + title2)
                file1.GetContentFile(survey_dir + title)
                file1.Delete()
    except ApiRequestError:
        print('file (probably) has already been downloaded!')
        pass

# Main functions for getting the sentinel images; here: only the directory for each survey is created
def sentinel_img_survey(img_p:str, csv_dir:str, sentinel_done:str, urban_radius:int,
                        rural_radius:int, MAX_CLOUD_PROBABILITY:int):
    '''Iterate over survey csvs and get Sentine-2 images for each cluster.
    
    Args:
        img_p(str):               Path to directory where Sentinel-2 images are stored
        csv_dir(str):               Path to directory where DHS csv files are stored
        sentinel_done(str):         Filepath for file to document for which surveys were are already completed
        urban_radius(int):          Radius around coordinates for Urban rgions in meter
        rural_radius(int):          Radius around coordinates for Rural regions in meter
        MAX_CLOUD_PROBABILITY(int): %
    '''
    if not os.path.exists(img_p):
        os.makedirs(img_p)

    if not os.path.isfile(sentinel_done):
        open(sentinel_done, 'a').close()

    csv_directory = os.listdir(csv_dir)
    img_directory = os.listdir(img_p)

    for file in csv_directory:
        if file.endswith('.csv'):
            filename = file[:file.rfind('.')]
            print(filename)
            # Check if survey is already done we skip this survey (sentinel_done file has to be edited manually)
            with open(sentinel_done) as f:
                if not filename in f.read():
                    survey_name = file[:file.rfind('.')]
                    survey_dir = os.path.join(img_p, survey_name)
                    if not os.path.exists(survey_dir):
                        os.makedirs(survey_dir)
                    file_dir = os.path.join(csv_dir, file)
                    get_survey_images(file_dir, survey_name, urban_radius, rural_radius, MAX_CLOUD_PROBABILITY)
                    # Add survey to txt file which stores all surveys which are done to avoid downloading them again if you reload the program
                    file1 = open(sentinel_done, "a")  # append mode
                    file1.write(file + "\n")
                    file1.close()
                    print(file, 'finished')



# sentinel_img_survey(img_p, csv_dir, sentinel_done, urban_radius, rural_radius, MAX_CLOUD_PROBABILITY)

def main(drive):
    download_local(img_p, drive)
    #list of available files
    for (dirrpath, dirrnames, filenames) in os.walk(img_p):
        print('available images', dirrpath, dirrnames, len(filenames))
        #only these folder are needed
    # print('d', filenames)
    with open(label_csv, 'r') as read_obj:
        # pass the file object to DictReader() to get the DictReader object
        dict_reader = DictReader(read_obj)
        # get a list of dictionaries from dct_reader
        clusters = list(dict_reader)
        # if clusters[0]['COUNTRY'].replace('_', ' ') == 'Democratic Republic of Congo':
        #     country_code = 'COD'
        # elif clusters[0]['COUNTRY'].replace('_', ' ') == 'Cote d\'Ivoire':
        #     country_code = 'CIV'
        # elif clusters[0]['COUNTRY'].replace('_', ' ') == 'Burkina Faso':

        #     country_code = 'BFA'
        # elif clusters[0]['COUNTRY'].replace('_', ' ') == 'Sierra Leone':
        #     country_code = 'SLE'
        # elif clusters[0]['COUNTRY'].replace('_', ' ') == 'Tanzania':
        #     country_code = 'TJK'
        #
        # else:
        #     country_code = country_code_map[clusters[0]['COUNTRY'].replace('_', ' ')]
    country_code = None
    final_clusters = []
    for cluster in clusters:
        # print('hi3')
        # print(cluster)
        if not cluster['TIF_name'] in filenames and int(float(cluster['DHSYEAR'])) >= min_year:
            final_clusters.append(cluster)

    pool = mp.Pool()
    len_clusters = len(final_clusters)
    t1 = time.time()
    t2 = time.time()
    for i, cluster in enumerate(final_clusters):
        # print('\n')
        res = pool.apply_async(get_image, args=(cluster, urban_radius, rural_radius, False, MAX_CLOUD_PROBABILITY,
                                                drive))
        if i and (not i % parallel_dls or not i % len_clusters):
            print(f'\rStarting {i} / {len_clusters}')
            #wait for one process to finish (it is random which gets started first!)
            #so not all downloads are started simultaneously (won't work!)
            #since it's random, there may be lots of processes still running, while new ones are started
            #you cannot see the old processes anymore, so it might take a long time until any of the new processes gets
            #actually started on gee (depends on parallel_dls)
            res.wait()
            # loc = res.get()
            # print('asd', loc)
            download_local(img_p, drive)
            rt30 = (time.time() - t2) / 3600
            t2 = time.time()
            rt = (t2 - t1)/3600
            eta = rt/i * (len_clusters - i)# + 1
            print(f'\rRunning time: {rt}h, running time for these {parallel_dls} tiles: {rt30}h, ETA: {eta}h or {eta/24} days')

    #wait 60 mins to ensure everything is downloaded - there are for sure more elegant ways to do this
    print(f'\rRunning time: {rt}h, ETA: {final_waiting_time}h ensuring everything is downloaded')
    time.sleep(60*60*final_waiting_time)
    print('downloading a last time')
    download_local(img_p, drive)
    print('finished')
    pool.close()
    # pool.join()
    # loc = get_image(cluster, urban_radius, rural_radius, False, MAX_CLOUD_PROBABILITY)
    # download files to local machine
    # if float(cluster["DHSCLUST"]) % 2 == 0:
    # download_local(os.path.join(img_p, survey_name))


if __name__ == "__main__":
    ###Initialize everything
    # authenticate once on the (remote) machine you are working on!!!
    # ee.Authenticate()
    print(f'Downloading urban tiles with a diameter of {urban_radius*2}m')
    print(f'Downloading rural tiles with a diameter of {rural_radius*2}m')

    #https://developers.google.com/earth-engine/cloud/highvolume used in
    # https://gorelick.medium.com/fast-er-downloads-a2abd512aa26
    # ee.Initialize(
    #     credentials=credentials,
    #     project=project,
    #     opt_url='https://earthengine-highvolume.googleapis.com'
    # )
    ee.Initialize(
        opt_url='https://earthengine-highvolume.googleapis.com')

    # ee.Initialize()
    print('initialized2')
    gauth = GoogleAuth()
    # only works on remote machine
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    #old stuff
    # gauth.LoadCredentialsFile("mycreds.txt")
    # drive = GoogleDrive(gauth)
    # for remote usage (does not work?!)
    # gauth.CommandLineAuth()
    main(drive)
