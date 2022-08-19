import os
import dhs_f
import fnmatch
import fiona
import csv
import sys

###Options
#since_year = 2000
#drop south africa, egypt, tunesia, morocco
#drop_countries = ['ZA', 'EG', 'TN', 'MA']
min_dhs_version = 6
#only extracts locations where these survey records are available #'HR' = household recode
# additional_file = 'HR'
###Paths
dhs_path = r"/mnt/datadisk/data/surveys/DHS_raw_data/"
projects_p = r"/mnt/datadisk/data/Projects/water/"
locations_f = projects_p + '/' + 'locations.csv'

dhs_d_all, country_d, data_file_types_d, typ_l, dhs_dirs_d = \
    dhs_f.load_dhs_data(dhs_path)
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
with open(locations_f, 'w') as csvfile:
    spamwriter = csv.writer(csvfile)
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
                if not header:
                    spamwriter.writerow(['country'] + list(props.keys()) + ['GEID'])
                    header = True
                #drop wrong coordinates:
                if props['LATNUM'] != 0 and props['LONGNUM'] != 0 and \
                        abs(props['LATNUM'] - point[1]) < 0.01 and abs(props['LONGNUM'] - point[0]) < 0.01:
                    # if props['DHSYEAR'] >= since_year:
                    row = [country_d[props['DHSCC']]] + list(props.values()) + [ge_n]
                    spamwriter.writerow(row)

                if props['LATNUM'] == 0 or props['LONGNUM'] == 0:
                    #print(props['LATNUM'], point[0], props['LONGNUM'], point[1])
                    zero_numbers.append(((props['LATNUM'], props['LONGNUM']), point))
                if abs(props['LATNUM'] - point[1]) > 0.01 or abs(props['LONGNUM'] - point[0]) > 0.01:
                    #print(props['LATNUM'], point[1], props['LONGNUM'], point[0])
                    wrong.append(((props['LONGNUM'], props['LATNUM']), point))

print('dropped locations with missing gps information (lat or long == 0):', len(zero_numbers), zero_numbers)
print('dropped unambiguous locations:', len(wrong), wrong)
