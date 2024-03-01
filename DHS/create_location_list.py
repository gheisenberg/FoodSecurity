import os
import pandas as pd
import redundant_code.dhs_f as dhs_f
import fnmatch
import fiona
import csv
import sys

###Options
#since_year = 2000
#drop south africa, egypt, tunesia, morocco
#drop_countries = ['ZA', 'EG', 'TN', 'MA']
min_dhs_version = False
#only extracts locations where these survey records are available #'HR' = household recode
# additional_file = 'HR'
###Paths
dhs_path = r"/mnt/datadisk/data/surveys/DHS_final_raw_data/"
dhs_extras_p = r"/mnt/datadisk/data/surveys/DHS_info/"
projects_p = r"/mnt/datadisk/data/Projects/water/inputs/"
locations_f = projects_p + '/' + 'final_all_locations_new.csv'
locations_f_spa = projects_p + '/' + 'final_SPA_locations.csv'

#gaul_locations_f = projects_p + '/' + 'all_locations_gaul2.csv'

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

dhs_d_all, country_d, data_file_types_d, typ_l, dhs_dirs_d = \
    dhs_f.load_dhs_data(dhs_path, dhs_extras_p=dhs_extras_p)
shp_pathes = {}

for id, types in dhs_d_all.items():
    #only extract shps where Household Recode data is available
    if 'GE' in types:# and additional_file in types:
        #drop South Africa, Egypt, Marocco,
        # if id[:2] not in drop_countries:
        if int(id[2]) >= min_dhs_version:
            for (dirrpath, dirrnames, filenames) in os.walk(dhs_dirs_d[id][types.index('GE')]):
                [path, ge_n] = os.path.split(dhs_dirs_d[id][types.index('GE')])
                for file in filenames:
                    if fnmatch.fnmatch(file, '*.shp'):
                        shp_pathes[ge_n] = dirrpath + '/' + file
print(len(shp_pathes), shp_pathes)

zero_numbers = []
wrong = []
header = False
header_spa = False
with open(locations_f, 'w') as csvfile:
    spamwriter1 = csv.writer(csvfile)
    with open(locations_f_spa, 'w') as csvfile2:
        spamwriter_spa = csv.writer(csvfile2)

        for ge_n, shp_path in shp_pathes.items():
            with fiona.open(shp_path) as shp:
                # schema of the shapefile
                for feature in shp:
                    props = feature['properties']
                    point = feature['geometry']['coordinates']
                    # if props['DHSID'] == 'GH201600000001':
                    #     sys.exit()
                    # props = dict(props)
                    try:
                        del props['OBJECTID']
                    except KeyError:
                        pass
                    if 'SPAID' in props:
                        spamwriter = spamwriter_spa
                        if not header_spa:
                            spamwriter.writerow(['country'] + list(props.keys()) + ['GEID', 'TIF_name'])
                            header_spa = True
                    else:
                        spamwriter = spamwriter1
                        if not header:
                            spamwriter.writerow(['country'] + list(props.keys()) + ['GEID', 'TIF_name'])
                            header = True

                    #drop wrong coordinates:
                    if props['LATNUM'] != 0 and props['LONGNUM'] != 0 and \
                            abs(props['LATNUM'] - point[1]) < 0.01 and abs(props['LONGNUM'] - point[0]) < 0.01:
                        # if props['DHSYEAR'] >= since_year:
                        try:
                            row = [country_d[props['DHSCC']]] + list(props.values()) + [ge_n, ge_n + props['DHSID'][-8:]]
                        except KeyError:
                            row = [country_d[props['DHSCC']]] + list(props.values()) + [ge_n,
                                                                                        ge_n + props['SPAID'][-8:]]
                        spamwriter.writerow(row)

                    elif props['LATNUM'] == 0 or props['LONGNUM'] == 0:
                        #print(props['LATNUM'], point[0], props['LONGNUM'], point[1])
                        zero_numbers.append(((props['LATNUM'], props['LONGNUM']), point))
                    elif abs(props['LATNUM'] - point[1]) > 0.01 or abs(props['LONGNUM'] - point[0]) > 0.01:
                        #print(props['LATNUM'], point[1], props['LONGNUM'], point[0])
                        wrong.append(((props['LONGNUM'], props['LATNUM']), point))
                    else:
                        print('something else went wrong')
                        sys.exit()
                        
print('dropped locations with missing gps information (lat or long == 0):', len(zero_numbers), zero_numbers)
print('dropped unambiguous locations:', len(wrong), wrong)

# #load gaul
# all_locations = pd.read_csv(locations_f)
# gaul_locations = pd.read_csv(gaul_locations_f)
# gaul_locations = gaul_locations[['DHSID', 'adm2_code', 'adm2_name',	'adm1_code', 'adm1_name', 'adm0_code',
#                                  'adm0_name']]
# all_locations = pd.merge(all_locations, gaul_locations, on='DHSID', how='outer')
# # print(all_locations)
# print(all_locations.head())
# print('\n\n')
# print(all_locations.describe())
# all_locations.to_csv(locations_f, index=False)
