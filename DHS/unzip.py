import os
import zipfile


zip_p = '/mnt/datadisk/data/surveys/DHS_zips_new'
dest_p = '/mnt/datadisk/data/surveys/DHS_final_raw_data'
add_zip_f_folder = True
for (dirrpath, dirrnames, filenames) in os.walk(zip_p):
    for nr, file in enumerate(filenames):
        if file[-4:].lower() == '.zip':
            f_p = dirrpath + '/' + file
            print('   unzipping', f_p, round(nr/len(filenames)*100, 1), '%' )
            try:
                with zipfile.ZipFile(f_p, 'r') as zip_ref:
                    if add_zip_f_folder:
                        zip_ref.extractall(dest_p + '/' + file[:-4])
                    else:
                        zip_ref.extractall(dest_p)
            #skip Error if zip file is corrupted
            except zipfile.BadZipFile:
                print('BadZipFile', f_p)
                continue
