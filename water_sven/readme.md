1. Download S2 files (refer to /Earth Engine/ee_sentinel.py)
2. Create labels (refer to /DHS/water_create_label.py)
3. Preprocess the data (refer to /DHS/water_preprocess.py)
4. If using own labels, create a csv file with the following columns:
    - 'GEID': the id of the DHS survey
    - 'DHSID': The ID of the clusters
    - 'LATNUM': latitude of the cluster
    - 'LONGNUM': longitude of the cluster
    - 'DHSYEAR': year of the survey
    - 'adm0_name': country name
    - 'adm1_name': region name
    - 'adm2_name': district name
    - 'households': number of households in the cluster
    - a column defining the split (train, val, test) or (split1-x, test for cross-validation)
    - a label column
5. Adjust the config file to your needs. It is highly accustomable! Refer to the config file for more information.
