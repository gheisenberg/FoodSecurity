Purpose: This code aims at creating a (csv-file) which connects HR-Files and GE-Files of the same survey to each other. The final file will store for each survey the HR-File name, the GE-file name, the year (derived from the GE-file) and the country abbreviation. The final csv-file is used later one for the CNN part.

Requirement: The code creating_water_source_files.ipynb was already executed and hence, the HR-files which do not have a corresponding GE-File are save in a seperate directory.

Process & Function:

1. Find corresponding GE-file: Via the function create_correspondes_file we at first try to find the corresponding GE for the current HR (iterating over all via for-loop).
Corresponds is given either if the survey names of the files are identically except of the 3-4 letter where the HR-Survey has as letters "HR" and the GE-file has as letters "GE" or when the year, the cluster number and the country is 
identical ( this is checked via check_year_country function). Please note: That HR may have multiple years given in the column "Year". However for corresponding year only one year value needs to be identical with the year of the GE-file.

2. Create Correspondence file: For each correspondence (via function create_correspondence_file) we extract the year & country (abbreviation) from the Ge-file (ge_year_country) and add as new element this new correspondence (with information about GE-file name and HR- file name, year and country) to a list. The list is later one transformed to a data frame and save as csv-file.S
