#!/bin/bash
set -euxo pipefail

GDRIVE_DIR="/home/shannon/Dokumente/Dokumente/studium/ASA/Projekt/SatelliteImage__GEE/correlation/GoogleDrive/"
GDRIVE_SYPHON="/run/media/shannon/TOSHIBA/sentinel"

DOWNLOAD_DIR=sentinel

cd $GDRIVE_DIR
# run grive
grive 

# copy files to siphon
rsync -a $GDRIVE_DIR $GDRIVE_SYPHON

# delete download dir
TIF_FILES_PATTERN=$GDRIVE_DIR/$DOWNLOAD_DIR/*.tif
if compgen -G "$TIF_FILES_PATTERN" > /dev/null; then
	rm $TIF_FILES_PATTERN
	grive
	# The Google Drive bin is now emptied every 10 minutes by an Apps Script EmptyTrashEvery10Min:
	# https://script.google.com/home/projects/1-RID38hCwOCVIL23vyExrp_q_vdAFkDYbr4jYZGwbdy40kAclD_vGlWa
	#cd /home/trossber/bin
	#source gdrive-venv/bin/activate
	#python3 empty-gdrive-trash.py
fi
