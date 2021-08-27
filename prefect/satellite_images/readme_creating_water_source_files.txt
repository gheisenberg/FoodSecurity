Purpose: 
This program creates as final product a CSV-file which contains all surveys, which have been conducted after 2012 and have a corresponding Geo-Date files to have access to the cluster locations.
The survey are shorten compared to the whole questionary for the CSV-file to solely the question regarding the water source, the cluster number, the residence type and the year.
The water source question is represented in the CSV-file by multiple columns, each one defining one possible water source. Please note,that the water sources given in the final CSV-file are a combination of multiple water sources given
in the originally survey questionaries as they may indicate the same source or are similar enough to join them. This is necessary as we want to use them as labels and too many labels are not suitable for the purpose of classification given the number of input data the NN can learn from.
Next to the final big CSV-file, the program also creates csv-files as an intermediate step for all HR-survey (HR = Household records) before they are joined to the final CSV-file. 

Requirement: Need to download all HR-Surveys (SPSS) in Zip-format from which you want to extract the Water source part of the questionairy in an own directory. Furthermore, to differentiate between HR-Surveys, which have a corresponding GeoData file containing the locations of the clusters, and HR-Surveys, which do not have those GeoData files, the program satellite_images_gee.ipynb has to be executed or at least the part which creates the csv-files.

Process & Functions:

1. Extract sav-files: Extract the SAV-Files from the ZIP-Data and save them in their own directory
    a)OPTIONAL def remove_FL(dir_name:str): Removes all ZIP-Data ending with FL.zip (as they are not relevant for our purpose)
    b) OPTIONAL def remove_all_except_HR(dir_name:str): Removes all ZIP-Data which are not Household Record (e.g. for the case that some ZIP files were wrongly downloaded)
    c) def extract_sav(dir_name:str, newpath:str): Is the main part for extracting the sav files; basically it runs through all zip files and jumps for each into the fct check_sav() where either the sav-file is directly extracted
    if one exists via get_sav() or if zip-file(s) within the current zip file exist(s), it jumps into def extract_bigger_zip() to proceed with the zip-files within the current zip-file which is the biggest one in size (and calls 
    with this one the function check_sav()). The sav-files are saved in their own directory.
2. Create the Water Source csv-files for each survey individually
    a) def create_csv(): Main function which iterates over all SAV-files and jumps for each into the function get_csv()
    b) get_csv(): Creates the csv-file by extracting from the SAV file the necessary information such as cluster number, year and water source. It joins them via a cross tab (dataframe). Please note: That for Ethopian HR survey (not
    GeoData files) the years are based on the Ethopian calendar. Hence, we need to translate them to the Gregorian calendar to be able to compare them later on with the GeoData files where the year is given already in Gregorian Dates.
    For this the function get_eth_to_gregorian() was implemented. 

3. Move all irrelevant CSV-files to seperate folders: As we are only interested in survey/csv files where the survey was conducted after 2012 and where a corresponding GeoData file exists (for the cluster locations), all other survey are
    moved to seperate folders.
    a) split_before_2013(): Moves all survey conducted before 2013 into a seperate folder 
    b) split_no_gps(): Moves all surveys which do not have corresponding GEO Data files to a seperate folder. We check for correspondence at first by checking if the survey names betweeen HR survey file and GeoData file are identical
    (by replacing HR with GE). If this is not the case, we further check if other indicators. This is required as for some survey the HR and GeoData files have different names. The indicators for checking for correspondences are same 
    year, same country (first two letters of the survey name) and same number of clusters. If those three indicators match, we assume that the files correspond to the same survey. If they do not match, we move the HR file to a seperate
    folder.

4. Creating a big CSV file: Firstly, we join all remaining CSV files to one single file with the fct create_single_csv(). To be able to figure out which row/cluster belongs to which survey a additional column containing the survey name
    is added. Secondly, via the function merge_columns_big_csv() we join some water source columns as many of them refer to similar sources. Which sources to join was decided in consulation with Sven & Yvonne.

