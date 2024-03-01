from pandas import DataFrame as df
import redundant_code.dhs_f as dhs_f

dhs_path = r"/mnt/datadisk/data/surveys/DHS_final_raw_data/"
dhs_extras_p = r"/mnt/datadisk/data/surveys/DHS_info/"
dhs_d_all, country_d, data_file_types_d, typ_l, dhs_dirs_d = \
    dhs_f.load_dhs_data(dhs_path, dhs_extras_p=dhs_extras_p)

dhs_d = {}
dhs_d['ID'] = ['Country_short', 'Country_long', 'vrsn'] + \
                  [data_file_types_d[typ] + " (" + typ + ")" for typ in typ_l]
print('typ', typ_l)
#print('\n\n', dhs_d['ID'])
#for k, v in dhs_d_all.items():
 #   print(k, v)
    
for k, v in dhs_d_all.items():
    ctry = k[:2]
    vrsn = k[2:]
    types = []
    for typ in typ_l:
        if typ in v:
            types.append(True)
        else:
            types.append(False)
    dhs_d[ctry + vrsn] = [ctry, country_d[ctry], vrsn] + types
    #print(dhs_d[ctry + vrsn])

dhs_df = df.from_dict(dhs_d)
dhs_df = dhs_df.transpose()
#print(dhs_df)
dhs_df.to_csv(dhs_extras_p + 'overview_final.csv')
