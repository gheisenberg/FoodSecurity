#!/bin/bash

# Fail early instead o silently ignoring errors;
#-e : exit immediately when a command fails (except conditional statements);-o pipefail: sets the exit code of a pipeline to that of the rightmost command to exit with a non-zero status,or to zero if all commands of the pipeline exit successfully.; -u:  treat unset variables as an error and exit immediately;-x option causes bash to print each command before executing it  
set -euxo pipefail

#Define directories (GDrive_DIR: Sync directory for Google Drive on local desktop; GDRIVE_SYPHON:unsynced directory where images are stored)
GDRIVE_DIR="/mnt/datadisk/shannon/get_sentinel/GoogleDrive/"
GDRIVE_SYPHON="/mnt/datadisk/shannon/get_sentinel/sentinel_images"
#Directory the images get downloaded to (within (un)synced directory directory next to export)
DOWNLOAD_DIR=sentinel


#Go to GDRIVE-DIR (snyced directory)and synchronize via grive with Google Drive
cd $GDRIVE_DIR
# run grive
grive 

# copy files to GDRIVE_SYPHON directory (unsynced one) from synced directory (GDRIVE_DIR); -a:  Do the sync preserving all filesystem attributes
rsync -a $GDRIVE_DIR $GDRIVE_SYPHON

# delete all tif-data in the snyced directory (folder sentinel) until none tif-data file is left and sync GDRIVE_DIR with Google Drive and hence, delete the images on the Drive as well 
TIF_FILES_PATTERN=$GDRIVE_DIR/$DOWNLOAD_DIR/*.tif
if compgen -G "$TIF_FILES_PATTERN" > /dev/null; then
	rm $TIF_FILES_PATTERN
	grive
	# The Google Drive bin is now emptied every 10 minutes by an Apps Script EmptyTrashEvery10Min:
fi
