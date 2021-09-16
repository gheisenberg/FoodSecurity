#!/usr/bin/env python
# coding: utf-8
import ee
ee.Initialize()
import ee
import geemap
import os
import functools
from zipfile import ZipFile
from dbfread import DBF
from dbfread import FieldParser
import shutil
import pandas as pd
from csv import DictReader
#from zdrive import Downloader
import time

#to dos:
#please add more comments what a function or a group of functions does
#generalize as a script with variables at the top
#ensure a whole year of data gets used for 2015 - use 2015-06-01 until 2016-07-01


# In[3]:


# Function to get a square around point of interest
# Rural : 5.5km Radius
# Urban : 2 km Radius
def bounding_box(loc, urban_rural, urban_radius, rural_radius):
    if urban_rural is 'U'or  urban_rural is 'u':
        size = urban_radius
    else:
        size = rural_radius

    intermediate_buffer = loc.buffer(size) #buffer radius, half your box width in m
    intermediate_box = intermediate_buffer.bounds() #Draw a bounding box around the circle
    return(intermediate_box)


# In[4]:


#Masking of clouds
def maskClouds(img, MAX_CLOUD_PROBABILITY):

    clouds = ee.Image(img.get('cloud_mask')).select('probability')
    isNotCloud = clouds.lt(MAX_CLOUD_PROBABILITY)
    return img.updateMask(isNotCloud)


# In[5]:


#Masking of edges
def maskEdges(s2_img):
    return s2_img.updateMask(s2_img.select('B8A').mask().updateMask(s2_img.select('B9').mask()))


# In[6]:


def get_image(cluster, survey_name, urban_radius, rural_radius, MAX_CLOUD_PROBABILITY):
    
    #Get images collections
    s2Sr = ee.ImageCollection('COPERNICUS/S2')
    s2Clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')

    #Get time span
    year_uncut = str(cluster["year"])
    year = year_uncut[:year_uncut.rfind('.')]
    if int(year)<2016:
        START_DATE = ee.Date('2015-06-01')
        END_DATE = ee.Date('2016-07-01')
    else:
        START_DATE = ee.Date(year+'-01-01')
        END_DATE = ee.Date(year+'-12-31')
    
   
    #Point of interest (longitude, latidude)
    lat_float = float(cluster["latidude"])
    lon_float = float(cluster["longitude"])                 
    loc = ee.Geometry.Point([lon_float, lat_float])
    #Region of interest
    region = bounding_box(loc, cluster['urban_rural'], urban_radius, rural_radius)

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
    s2CloudMasked = s2CloudMasked.select(['B1','B2','B3', 'B4', 'B5', 'B6', 'B7', 'B8','B8A', 'B9', 'B10'                                         ,'B11','B12']).clip(region)
    #Saving location/directory
    #out_dir = os.path.join(survey_dir, cluster["ID-cluster"]+'.tif')
    #geemap.ee_export_image(s2CloudMasked, filename=out_dir, scale=10)
    filename = cluster["ID-cluster"]
    filename = filename.replace(filename[:6], survey_name)
    task = ee.batch.Export.image.toDrive(s2CloudMasked, description = filename, folder = 'sentinel', scale = 10)
    task.start()
    print('Created', filename)
    return loc


# In[7]:


# Extract solely the dbf-file from the zip and save them into a seperate folder
def get_dbf(listOfFileNames, newpath, zipObject):

    for fileName in listOfFileNames:
        if fileName.endswith('.dbf') or fileName.endswith('.DBF'):
            # Extract a single file from zip
            zipObject.extract(fileName, newpath)
            
#If within a zip folder other zip folders included take/work with the the zip folder which is bigger            
def extract_bigger_zip(zip_dir,filenames,newpath):
    zips_dir = zip_dir[:zip_dir.find('.')]
    if not os.path.exists(zips_dir):
        os.makedirs(zips_dir)
    with ZipFile(zip_dir, 'r') as zipObj:
    #Extract all the contents of zip file in different directory
        zipObj.extractall(zips_dir)

    big_size = 0
    big_zip = None
    for item in filenames:
        if item.endswith('.zip') or item.endswith('.ZIP'):
            file_dir = os.path.join(zips_dir, item)
            curr_size= os.stat(file_dir).st_size
            if curr_size >= big_size:
                big_size = curr_size
                big_zip = file_dir
    check_dbf(big_zip,newpath)    
    
#Second mainpart for single zip file        
def check_dbf(zip_dir, newpath):
    big_zip = None
    with ZipFile(zip_dir, 'r') as zipObject:
        listOfFileNames = zipObject.namelist()

        if any((element.endswith('.dbf') or element.endswith('.DBF')) for element in listOfFileNames):
            get_dbf(listOfFileNames, newpath, zipObject)
        elif any((element.endswith('.zip') or element.endswith('.ZIP')) for element in listOfFileNames):
            extract_bigger_zip(zip_dir, listOfFileNames, newpath)  

#Delete all directories which may be created during the process to figure out which zip folder is bigger             
def delete_zip_folders(dir_name,newpath):
    list_subfolders_with_paths = [f.path for f in os.scandir(dir_name) if f.is_dir()]
    for folder in list_subfolders_with_paths:
        if not folder == newpath:
            shutil.rmtree(folder)
            
#Main part-> runs through all zip files in directory  
def extract_dbf(dir_name, newpath):
    #Create folder for DBF files
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    folder = os.listdir(dir_name)
    #Extract dbf file (if existing) from each zip file
    for item in folder:
        if item.endswith('.zip') or item.endswith('.ZIP'):
            zip_dir = os.path.join(dir_name, item)
            check_dbf(zip_dir, newpath)
            
    delete_zip_folders(dir_name,newpath)
    


# In[8]:


#Needed as a file may contain \x00\ which the standard libary is not able to process, hence an additional class based on the following issue https://github.com/olemb/dbfread/issues/20 is used 

class MyFieldParser(FieldParser):
    def parseN(self, field, data):
        data = data.strip().strip(b'*\x00')  # Had to strip out the other characters first before \x00, as per super function specs.
        return super(MyFieldParser, self).parseN(field, data)

    def parseD(self, field, data):
        data = data.strip(b'\x00')
        return super(MyFieldParser, self).parseD(field, data)
    
    
#Create single CSV file    
def get_csv(dbf_path, dir_csv):

    clusters = list()
    filename = os.path.basename(dbf_path[:dbf_path.rfind('.')])
    if not filename.endswith('SR'):
        table = DBF(dbf_path, parserclass=MyFieldParser)
        for record in table:
        
            cluster = {"ID-survey": filename, "ID-cluster" : record['DHSID'], "cluster": record['DHSCLUST'],                        "year": record["DHSYEAR"],"urban_rural": record['URBAN_RURA'], "latidude": record["LATNUM"],                      "longitude": record['LONGNUM']}
            clusters.append(cluster)
    
        clust_df = pd.DataFrame(clusters)
    
        export_name = filename+'.csv'
        export_dir = os.path.join(dir_csv, export_name)

        clust_df.to_csv(export_dir)
    else:
        print(filename)


# In[9]:


#Create CSV files for each survey (with dbf file); Extract information from dbf file   
def create_csv(dir_dbf, dir_csv):
    if not os.path.exists(dir_csv):
        os.makedirs(dir_csv)

    directory = os.listdir(dir_dbf)
    for file in directory: 
        #print("This is the file", file)
        dbf_path = os.path.join (dir_dbf, file)
        get_csv(dbf_path, dir_csv)
    return 


# In[10]:


#Move all csv-files from surveys which took place before 2013 to a seperate folder
def before_2013 (export_path, before_2013):
    
    if not os.path.exists(before_2013):
        os.makedirs(before_2013)

    directory = os.listdir(export_path)    
    for file in directory:
        #print(file)
        if file.endswith('.csv'):
            csv_file = os.path.join(export_path, file)
            survey_year = pd.read_csv(csv_file, usecols = ['year'])

            if survey_year['year'].max()< 2013:
                new_path = os.path.join(before_2013, file)
                os.rename(csv_file, new_path)


# In[11]:



def get_survey_images(file_dir, survey_name, urban_radius, rural_radius, MAX_CLOUD_PROBABILITY):
    with open(file_dir, 'r') as read_obj:
    # pass the file object to DictReader() to get the DictReader object
        dict_reader = DictReader(read_obj)
    # get a list of dictionaries from dct_reader
        clusters = list(dict_reader)
        
    for cluster in clusters:
        loc = get_image(cluster, survey_name, urban_radius, rural_radius, MAX_CLOUD_PROBABILITY)
        time.sleep(45)
'''   
def download_local(survey_dir):
    output_directory = survey_dir
    d = Downloader()

    # folder which want to download from Drive
    folder_id = '1ST67vgoNlfuClI-zPlEp4F38JsnaM441'
    d.downloadFolder(folder_id, destinationFolder=output_directory)
'''    
#Main functions for getting the sentinel images; here: only the directory for each survey is created     
def sentinel_img_survey(img_dir, csv_dir, sentinel_done, urban_radius, rural_radius, MAX_CLOUD_PROBABILITY):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        
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
                    #download_local(survey_dir)
                    print(file, 'finished')


# In[12]:


#Main Part

#Parameter
urban_radius = 1000
rural_radius = 5000
MAX_CLOUD_PROBABILITY = 20


#Paths
zip_dir = "/home/shannon/Dokumente/Dokumente/studium/ASA/Projekt/SatelliteImage__GEE/correlation/GPS_Data"
dbf_dir = os.path.join(zip_dir, "dbf_files")
csv_dir = os.path.join(zip_dir, "gps_csv")
before_2013_dir = os.path.join(csv_dir, "before_2013")
#if directly to local computer
#img_dir = os.path.join(csv_dir, "tif_data")
img_dir = '/run/media/shannon/TOSHIBA/Sentinel'
sentinel_done = os.path.join(zip_dir, "sentinel_done.txt")


#Functions
extract_dbf(zip_dir, dbf_dir)
print('Extracted dbf files')
create_csv(dbf_dir, csv_dir)
print('created csv data')
before_2013(csv_dir, before_2013_dir)
print('Moved all surveys from before 2013 into seperate folder')
sentinel_img_survey(img_dir, csv_dir, sentinel_done, urban_radius, rural_radius, MAX_CLOUD_PROBABILITY)


# In[ ]:




