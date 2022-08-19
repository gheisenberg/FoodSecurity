import os
import csv
import fnmatch
import numpy as np


def load_dhs_data(dhs_path, return_paths=False):
    for (dirrpath, dirrnames, filenames) in os.walk(dhs_path):
        print(dirrpath, dirrnames, filenames)
        #only these folder are needed
        break

    dirrnames.sort()

    ctry_s = set()
    typ_s = set()
    file_format_s = set()
    file_format_d = {'FL': 'flat', 'SV': 'SPSS'}

    dhs_d_all = {}
    dhs_dirs_d = {}
    for dirr in dirrnames:
        ctry = dirr[:2]
        ctry_s.add(ctry)
        typ = dirr[2:4]
        typ_s.add(typ)
        vrsn = dirr[4:5]
        # vrsnb is release version (if corrections have been made) - needs to be seperated for matching purposes
        vrsnb = dirr[5:6]
        try:
            if type(int(vrsnb)) == int:
                survey_n = 1
        except ValueError:
            if vrsnb in 'ABCDEFGH':
                survey_n = 2
            elif vrsnb in 'IJKLMNOPQ':
                survey_n = 3
            elif vrsnb in 'RSTUVWXYZ':
                survey_n = 4
            else:
                raise IOError("this version should not exist: " + vrsnb)
        survey_n = str(survey_n)
        file_format = dirr[6:]
        file_format_s.add(file_format)
        k = ctry + vrsn + survey_n
        #only select shps and spss data
        if file_format == 'SV':
            try:
                dhs_d_all[k].append(typ)
                dhs_dirs_d[k].append(dirrpath + dirr)
            except KeyError:
                dhs_d_all[k] = [typ]
                dhs_dirs_d[k] = [dirrpath + dirr]
        else:
            if typ == 'GE':
                try:
                    dhs_d_all[k].append(typ)
                    dhs_dirs_d[k].append(dirrpath + dirr)
                except KeyError:
                    dhs_d_all[k] = [typ]
                    dhs_dirs_d[k] = [dirrpath + dirr]

    dhs_d_all = dict(sorted(dhs_d_all.items()))
    # print(dhs_d_all)


    ctry_l = list(ctry_s)
    ctry_l.sort()
    typ_l = list(typ_s)
    typ_l.sort()


    country_d = {}
    input_file = csv.reader(open(r"/mnt/datadisk/data/surveys/DHS_raw_data/country_codes.csv"), delimiter='\t')
    for row in input_file:
        country_d[row[0]] = row[1]

    data_file_types_d = {}
    input_file = csv.reader(open(r"/mnt/datadisk/data/surveys/DHS_raw_data/data_file_types.csv"), delimiter='\t')
    for row in input_file:
        data_file_types_d[row[0]] = row[1]

    return dhs_d_all, country_d, data_file_types_d, typ_l, dhs_dirs_d


def water_source_renaming(row):
    print(row)
    if type(row) is float:
        if np.isnan(row['source of drinking water']):
            return_v = np.NaN
        else:
            raise Exception('This should not happen: float but no np.NaN', row['source of drinking water'])
    else:
        if 'unprotected' in row['source of drinking water']:
            if 'well' in row['source of drinking water']:
                return_v = 'unprotected well'
            elif 'spring' in row['source of drinking water']:
                return_v = 'unprotected spring'
        elif 'protected' in row['source of drinking water']:
            if 'well' in row['source of drinking water']:
                return_v = 'protected well'
            elif 'spring' in row['source of drinking water']:
                return_v = 'protected spring'
        elif 'open' in row['source of drinking water'] and 'well' in row['source of drinking water']:
            return_v = 'unprotected well'
        # elif 'well' in row['source of drinking water']:
        #     return_v = 'unprotected well'
        elif 'sachet' in row['source of drinking water'] or 'bag water' in row['source of drinking water'] \
                or 'water in plastic bag' in row['source of drinking water']:
            return_v = 'sachet'
        elif 'bottled' in row['source of drinking water']:
            return_v = 'bottled'
        elif 'borehole' in row['source of drinking water']:
            return_v = 'Tube well or borehole'
        elif 'truck' in row['source of drinking water'] or 'cart' in row['source of drinking water'] or \
                'vendor' in row['source of drinking water'] or 'tanker' in row['source of drinking water'] or \
                'bicycle' in row['source of drinking water'] or 'motorcycle' in row['source of drinking water']:
            return_v = 'street vendor'
        elif 'river' in row['source of drinking water'] or 'gravity flow scheme' in row['source of drinking water']:
            return_v = 'stream'
        elif 'dam' in row['source of drinking water'] or 'pond' in row['source of drinking water']:
            return_v = 'standing water'
        elif "neighbor's tap" in row['source of drinking water'] \
            or "piped to neighbour's house" in row['source of drinking water']  \
                or "piped from the neighbor" in row['source of drinking water']\
                or "neighbour's tap" in row['source of drinking water']\
                or "private tap/neighbor" in row['source of drinking water']\
                or "neighbor's house" in row['source of drinking water']:
            return_v = 'piped to neighbor'
        elif "neighborhood" in row['source of drinking water']:
            return_v = 'unprotected well'
        elif "spring" == row['source of drinking water']:
            return_v = "unprotected spring"
        elif "public fountain" in row['source of drinking water']:
            return_v = "other"
        else:
            return_v = row['source of drinking water']
        # print(row['source of drinking water'], return_v)
    return return_v

#
# def water_source_weighting(row):
#     if row['source of drinking water'] == "Tube well or borehole":
#
#     186580
#     public
#     tap / standpipe
#     137048
#     unprotected
#     well
#     107751
#     stream
#     92828
#     protected
#     well
#     84582
#     piped
#     to
#     yard / plot
#     77720
#     unprotected
#     spring
#     52660
#     protected
#     spring
#     49526
#     piped
#     into
#     dwelling
#     45018
#     piped
#     to
#     neighbor
#     24817
#     sachet
#     14789
#     street
#     vendor
#     9971
#     rainwater
#     8800
#     bottled
#     5508
#     other
#     5108
#     standing
#     water
#     445
