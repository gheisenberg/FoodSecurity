requirments: grive, local synchronization folder of google drive
copy .timer and .service into /home/.config/systemd/user/
ausführen (ergänzen Shannon)
change paths in gdrive-siphon.sh
you might need to run it with "grive -a" first to get access to google drive
to only synchronize one folder use "grive -s $foldername"
#Still Problems with weird file name/file id (it does not use the displayed name but an internal id for the copied file)


#New Python solution:
#Make drive available on ubuntu: https://www.how2shout.com/how-to/install-use-google-backup-and-sync-ubuntu-linux-command-terminal-html.html
#Still Problems with weird file name/file id
