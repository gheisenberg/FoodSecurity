A. Purpose: The program downloads Sentinel2 Images via the Google Earth Engine for DHS Surveys (2013-until now) which have GeoData files (GeoData files are indicated by 'GE' on the 3-4 position of the name).

B. Requirement: - Zip-Files of the GeoData has to be downloaded manually and has to be stored within a single directory ('zip_dir'). 
             - The code is implemented in Python 3.8.10. 
             - Goggle Account needed (GoogleEarthEngine and GoogleDrive).

C. Procedure & Functions (ordered by usage): 

    1. Extracting DBF: At the begin the dbf files are extracted from the zip files and a stored in a seperate directory. Note: The dbf-file is used as it contains all necessary information (location, cluster number, residence type)

        a) def extract_dbf(dir_name:str, newpath:str): Main function of this part; creates folder for the dbf files ('newpath')
        b) def check_dbf(zip_dir:str, newpath:str): Checks if a dbf-file is given in the zip-file (->jump into get_dbf) and if not, if a zip-file or more a existing (-> jump into extract_bigger_zip)
        c) def extract_bigger_zip(zip_dir:str,filenames:str,newpath:str): Takes the biggest zip file within a zipfile w.r.t to the size and jumps back into check_dbf
        d) def get_dbf(listOfFileNames:list, newpath:str, zipObject:zipfile.ZipFile): Extract dbf-file and store it into the newly created folder ('newpath')
    2. Create CSV-file: Based on the dbf-files csv-files are created for each survey which has a dbf-file. The csv-file contains information about location, survey, cluster number and residence type
        a) def create_csv(dir_dbf:str, dir_csv:str): Main function of this part; creates folder for the csv-files ('dir_csv')
        b) def get_csv(dbf_path:str, dir_csv:str): Extracts information from the dbf-file and saves the dataframe as csv-file in the new directory. The name of each file corresponds to the respectively survey name
    3. Split into before and after 2013: As the Sentinel 2 images aquistion started in May 2015, we decided to only download images belonging to surveys which took place 2013 and later to ensure that satellite images reflect the given
        answers to the survey questions.
        a) def before_2013 (export_path:str, before_2013:str): All surveys which have as date 2012 and earlier are moved to a seperate directory
    4. Get sentinel images for each survey: This is the main part of the program. The sentinel images are downladed for each survey.Thereby the center points of the images correspond to the cluster locations of the surveys retrieved 
        from the dbf files and saved in the csv file.
        a)def sentinel_img_survey(img_dir:str, csv_dir:str, sentinel_done:str, urban_radius:int, rural_radius:int, MAX_CLOUD_PROBABILITY:int): Main function of this part; if a survey is done (all images downloaded) the survey name is   
        added to the txt file sentine_done.txt such that if the program is started again the survey will be skipped and the images not downloaded again to minimze running time
        b) def get_survey_images(file_dir:str, survey_name:str, urban_radius:int, rural_radius:int, MAX_CLOUD_PROBABILITY:int): Reads out the clusters of a survey and jumps for each cluster into get_image(). Note: time.sleep() is required
        to avoid that the GEE stops the program has it has to proceed to many queries at the same time
        c) def get_image(cluster:dict, survey_name:str, urban_radius:int, rural_radius:int, MAX_CLOUD_PROBABILITY:int): Downloads a final sentinel image. For this purpose all satellite images for a defined time frame (1 year) and a 
        defined region are collected for the Sentinel2 Image collection and for the Sentinel2 Image collection which states the cloud probability for each pixel. The defined time frame is usually the year the survey was conducted,
        except if it was before 2016, then the time frame is predefined as the Sentinel2 images aquistion started not before May 2015. The region - a square- has as center point the location given by the cluster and a width/length
        defined by a predefined value, differently for urban and rural region [function bounding_box] . The collected images are used to gain the final image. Each of the collected images are masked, meaning that the pixels are "zeroed"
        if the probablity for a cloud is higher than a predfined threshold (MAX_CLOUD_PROBABILITY) [function maskEdges and maskClouds]. After masking the images are joined in such a way that as many zero pixel are replaced by non-zero
        pixels for the final image which is then downloaded via GEE and saved to your Google Drive.
        Note: To avoid running out of memory space of your Google Drive you may use the gdrive programs (also available in the github repository).
    


 
