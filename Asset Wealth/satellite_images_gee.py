#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 10:49:05 2021

@author: shannon
"""
import ee
import geemap
import os
import functools
from zipfile import ZipFile
import shutil
import pandas as pd
from csv import DictReader
#from zdrive import Downloader
import time
#ee.Authenticate()
ee.Initialize()


# Function to get a square around point of interest
# Rural : 10 km Radius
# Urban : 2 km Radius
def bounding_box(loc, urban_rural, urban_radius, rural_radius):
    if urban_rural == 'U'or  urban_rural == 'u':
        size = urban_radius
    else:
        size = rural_radius

    intermediate_buffer = loc.buffer(size) #buffer radius, half your box width in m
    intermediate_box = intermediate_buffer.bounds() #Draw a bounding box around the circle
    return(intermediate_box)


#Masking of clouds
def maskClouds(img, MAX_CLOUD_PROBABILITY):

    clouds = ee.Image(img.get('cloud_mask')).select('probability')
    isNotCloud = clouds.lt(MAX_CLOUD_PROBABILITY)
    return img.updateMask(isNotCloud)


#Masking of edges
def maskEdges(s2_img):
    return s2_img.updateMask(s2_img.select('B8A').mask().updateMask(s2_img.select('B9').mask()))


def get_image(cluster, survey_name, urban_radius, rural_radius, MAX_CLOUD_PROBABILITY):
    
    #Get images collections
    s2Sr = ee.ImageCollection('COPERNICUS/S2')
    s2Clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')

    #Get time span
    year_uncut = str(cluster["SURVEY_YEAR"])
    year = year_uncut[:year_uncut.rfind('.')]
    if int(year)<2016:
        START_DATE = ee.Date('2015-06-01')
        END_DATE = ee.Date('2016-07-01')
    else:
        START_DATE = ee.Date(year+'-01-01')
        END_DATE = ee.Date(year+'-12-31')
    
   
    #Point of interest (longitude, latidude)
    lat_float = float(cluster["LATNUM"])
    lon_float = float(cluster["LONGNUM"])
    loc = ee.Geometry.Point([lon_float, lat_float])
    #Region of interest
    region = bounding_box(loc, cluster['URBAN_RURA'], urban_radius, rural_radius)

    # Filter input collections by desired data range and region.
    #criteria = ee.Filter.And(ee.Filter.bounds(region), ee.Filter.date(START_DATE, END_DATE))
    #s2Sr = s2Sr.filter(criteria).map(maskEdges)
    #s2Clouds = s2Clouds.filter(criteria)
    s2Sr = s2Sr.filterBounds(region).filterDate(START_DATE, END_DATE).map(maskEdges)
    s2Clouds = s2Clouds.filterBounds(region).filterDate(START_DATE, END_DATE)

    # Join S2 with cloud probability dataset to add cloud mask.
    s2SrWithCloudMask = ee.Join.saveFirst('cloud_mask').apply(
      primary =  s2Sr, 
      secondary = s2Clouds, 
      condition = ee.Filter.equals(
          leftField =  'system:index', rightField = 'system:index') 
        )

    maskCloudsWithProb = functools.partial(maskClouds, MAX_CLOUD_PROBABILITY = MAX_CLOUD_PROBABILITY)
    s2CloudMasked = ee.ImageCollection(s2SrWithCloudMask).map(maskCloudsWithProb).median()
    s2CloudMasked = s2CloudMasked.select(['B1','B2','B3', 'B4', 'B5', 'B6', 'B7', 'B8','B8A', 'B9', 'B10'\
                                         ,'B11','B12']).clip(region)
    #Saving location/directory
    #out_dir = os.path.join(survey_dir, cluster["ID-cluster"]+'.tif')
    #geemap.ee_export_image(s2CloudMasked, filename=out_dir, scale=10)
    filename = survey_name
    task = ee.batch.Export.image.toDrive(s2CloudMasked, description = filename, folder = 'sentinel', scale = 10)
    task.start()
    print('Created', filename)
    return loc



def get_survey_images(file_dir, survey_name, urban_radius, rural_radius, MAX_CLOUD_PROBABILITY):
    with open(file_dir, 'r') as read_obj:
    # pass the file object to DictReader() to get the DictReader object
        dict_reader = DictReader(read_obj)
    # get a list of dictionaries from dct_reader
        clusters = list(dict_reader)
        
    for cluster in clusters:
        loc = get_image(cluster, survey_name, urban_radius, rural_radius, MAX_CLOUD_PROBABILITY)
        time.sleep(45)

def download_local(survey_dir):
    output_directory = survey_dir
    d = Downloader()

    # folder which want to download from Drive
    folder_id = '1mUgpprO70dFtfu5yyHLVfsjvh0wcRhUw'
    d.downloadFolder(folder_id, destinationFolder=output_directory)

#Main functions for getting the sentinel images; here: only the directory for each survey is created     
def sentinel_img_survey(img_dir, csv_dir, sentinel_done, urban_radius, rural_radius, MAX_CLOUD_PROBABILITY):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        
    if not os.path.isfile(sentinel_done):
        open(sentinel_done, 'a').close()
        
    csv_directory = os.listdir(csv_dir)
    img_directory = os.listdir(img_dir)
    
    for file in csv_directory:
        if file.endswith('.csv'):
            filename  = file[:file.rfind('.')]
            #Check if survey is already done we skip this survey (sentinel_done file has to be edited manually)
            with open(sentinel_done) as f:
                if not filename in f.read():
                    survey_name = file[:file.rfind('.')]
                    survey_dir = os.path.join(img_dir, survey_name)
                    if not os.path.exists(survey_dir):
                        os.makedirs(survey_dir)
                    file_dir = os.path.join(csv_dir, file)
                    get_survey_images(file_dir, survey_name, urban_radius, rural_radius, MAX_CLOUD_PROBABILITY)
                    download_local(survey_dir)
                    #Add survey to txt file which stores all surveys which are done to avoid downloading them again if you reload the program
                    file1 = open(sentinel_done,"w")#write mode
                    file1.write(file+"\n")
                    file1.close()
                    print(file, 'finished')
                    
                    
                    
#Main Part

#Parameter
urban_radius = 2000 # meter
rural_radius = 10000 # meter
MAX_CLOUD_PROBABILITY = 20 # %


#Paths
#Directory of originally zip_files containing the dbf-files
zip_dir = "/home/stoermer/Sentinel"
#Directory where csv files are stored
csv_dir = os.path.join(zip_dir, "gps_csv")
#if directly to local computer
#img_dir = os.path.join(csv_dir, "tif_data")
#Directory where the final survey zips containing the sentinel images are stores
img_dir = '/home/stoermer/Sentinel/Sentinel_zip'
#Directory to txt files which contains all surveys where the images were already retrieved
sentinel_done = os.path.join(zip_dir, "sentinel_done.txt")

#Functions
sentinel_img_survey(img_dir, csv_dir, sentinel_done, urban_radius, rural_radius, MAX_CLOUD_PROBABILITY)
