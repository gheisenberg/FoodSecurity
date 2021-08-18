import os
import shutil
#import sl timer

g_drive = "/run/user/1003/gvfs/google-drive:host=gmx.de,user=gnila"
s2_drive = g_drive + '/sentinel' 
print(s2_drive)
s2_dir = "/mnt/datadisk/sciebo/DIS22/Data_Acquisition/Sentinel2"

file_l = os.listdir(s2_drive)
print(file_l)
for fn in file_l:
   fp = s2_drive + '/' + fn
   shutil.move(fp, s2_dir + '/' + fn)
