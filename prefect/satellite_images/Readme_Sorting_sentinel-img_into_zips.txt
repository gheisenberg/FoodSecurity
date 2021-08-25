A. Purpose: This code aims at grouping the Sentinel Images collected beforehand via the program satellite_images_gee.ipynb w.r.t to their survey. Each group belongs to one survey and is saved as a Zipfile.

B. Requirements: This code assumes that satellite_images_gee was already executed and hence, the tif-files/sentinel images and the corresponding survey folders exist. Furthermore, it is assumed that the satellite images are retrieved from your Google Drive to a local directory. The Python Version used is 3.8.10.

C. Process & Functions:
    1. Moving images into their respective folders
        a) def move_imgs_to_folder(tif_dir:str,folders_dir:str):  Main function of this part; Jumps for each folder, hence each survey, into move_surveys_tifs().
        Note: the creation of the list  folders uses not the predfined function os.listdir() as it wants to avoid searching through hiddden folder which would be added to the list with this function. Hence, we define our own function
        def listdir_nohidden(path:str) to only add visible folders (the survey folders) to the list. This minimizes the running time of the program.  
        b)def move_surveys_tifs(folder:str, folder_dir:str, tif_dir:str): This function runs through all tif files stored in the tif_dir [e.g. used for storing the images retrieved from Google Drive via gdrive programs] and moves every
        tif file, where the name of the current survey is given within the filename of the tif file, to this respective survey folder/directory in the folder directory [folder_dir]
    2. Transform the directories/survey folders into zip files: This minimizes the required memory capacity.
        a) def zip_and_delete_folders(folders_dir:str): Firstly, for each survey directory create a zipfile containing the tif images of the survey. The name of the zipfile is the surveyname. Secondly, delete the folders/directories.

  


