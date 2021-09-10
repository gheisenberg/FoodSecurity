SystemD

Purpose: As you are downloading our satellite images to your Google Drive, it is necessary to empty the Google Drive frequently (if not using the Premium Version), because the storage space is limited to 15 GB.
This set of scripts, named gdrive-siphon.*, downloads on a reguarly basis the satellite images from your Google Drive to a synchronized local directory. From this local directory we copy the sentinel images to a local directory (not synced with the Google Drive) and delete all files within the synced directory. Now, the synced directory snychronizes with the Google Drive and hence, all files there will be deleted as well. The Google drive trash bin is also emptied 
to avoid that the storage space is full.



Requirements:
* install grive2 (https://github.com/vitalif/grive2)
* setup two directories:
  * one to sync with the online G Drive
  * another one, where all data from the G Drive is stored
* linux operating system with systemd

Systemd is a deamon managing services needed for normal computer use (like printing service, audio, bluetooth, network manager). 
It allows running scripts periodically and is therefore used for this setup.

Parts:
A. gdrive-siphon.timer
B. gdrive-siphon.service
C. gdrive-siphon.sh
D. Google App Script to empty the trash bin


## gdrive-siphon.timer
* place file under ~/.config/systemd/user/
* periodically runs the gdrive-siphon.service


## gdrive-siphon.service
* place file under ~/.config/systemd/user/
* is run by the timer
* executes the shell script containing all the logic (gdrive-siphon.sh)
* location of gdrive-siphon.sh needs to be defined here

## gdrive-siphon.sh
* contains the logic of the system
* place at any location and adjust the path to it in gdrive-siphon.service




## Google App Script to empty trash bin
```
/* 
Code taken from:
https://stackoverflow.com/questions/32749289/automatically-delete-file-from-google-drive-trash
*/
function createTimeDrivenTriggers() {
  ScriptApp.newTrigger('emptyThrash')
      .timeBased()
      .everyMinutes(10)
      .create();
}

function emptyThrash()
{
  Drive.Files.emptyTrash();
}
```



