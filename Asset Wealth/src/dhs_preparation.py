# Imports
import pyreadstat
import requests
import json
import pandas as pd
import os
import geopandas as gpd

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing

floor_recode_df, meta = pyreadstat.read_dta('/mnt/datadisk/data/surveys/asset/dhs_data/recode_tables/floor_recode.dta')
toilet_recode_df, meta = pyreadstat.read_dta('/mnt/datadisk/data/surveys/asset/dhs_data/recode_tables/toilet_recode.dta')
water_recode_df, meta = pyreadstat.read_dta('/mnt/datadisk/data/surveys/asset/dhs_data/recode_tables/water_recode.dta')

floor_recode_dict = floor_recode_df.set_index('floor_code').to_dict()['floor_qual']
toilet_recode_dict = toilet_recode_df.set_index('toilet_code').to_dict()['toilet_qual']
water_recode_dict = water_recode_df.set_index('water_code').to_dict()['water_qual']
water_recode_dict[63] = 4


class DHS_preparation():
    def __init__(self,
                 floor_recode: dict,
                 toilet_recode: dict,
                 water_recode: dict,
                 country_code_dict: dict,
                 features: list,
                 info: list,
                 dhs_survey_path: str,
                 wealth_path: str,
                 shape_path: str,
                 geo_wealth_path: str,
                 sustainlab_group_file: str
                 ):
        '''        
        Args:
            floor_recode (dict): Mapping for floor type to int.
            toilet_recode (dict): Mapping for toilet type to int.
            water_recode (dict): Mapping for water type to int.
            country_code_dict (dict): Mapping of DHS country code to country name.
            features (list): Features (column names) to include for Asset Wealth calculation.
            info (list): Names of columns that include information about the cluster which is not required for Asset Wealth calculation (e.g. Household ID).
            dhs_survey_path (str): Path to DHS survey data.
            wealth_path (str): Output path for wealth files.
            shape_path (str): Path to DHS geoinformation.
            geo_wealth_path (str): Output path for final wealth files including geoinformation.
            sustainlab_group_file (str): Path to sustainlab group csv file.
        '''
        self.floor_recode = floor_recode
        self.toilet_recode = toilet_recode
        self.water_recode = water_recode
        self.country_code_dict = country_code_dict
        self.features = features
        self.info = info
        self.dhs_survey_path = dhs_survey_path
        self.wealth_path = wealth_path
        self.shape_path = shape_path
        self.geo_wealth_path = geo_wealth_path
        self.sustainlab_group_file = sustainlab_group_file

    def recode_and_format_dhs(self, filename: str):
        '''Recode DHS survey data and calculate Asset Wealth.

        Args:
            filename: Filename of DHS survey csv
        '''
        # load sas data to pandas dataframe
        survey_df, meta = pyreadstat.read_sas7bdat(os.path.join(self.dhs_survey_path, filename))

        # select only required columns and rename
        survey_df = survey_df[
            ['HHID', 'HV000', 'HV001', 'HV002', 'HV007', 'HV009', 'HV025', 'HV201', 'HV205', 'HV206', 'HV207',
             'HV208', 'HV209', 'HV211', 'HV212', 'HV213', 'HV216', 'HV221', 'HV243A']]
        survey_df = survey_df.rename(columns={'HV001': 'CLUSTER', 'HV002': 'HOUSEHOLD',
                                              'HV007': 'YEAR', 'HV009': 'MEMBERS', 'HV025': 'URBAN_RURAL',
                                              'HV201': 'WATER_SOURCE',
                                              'HV205': 'TOILET_TYPE', 'HV206': 'ELECTRICITY', 'HV207': 'RADIO',
                                              'HV208': 'TV',
                                              'HV209': 'FRIDGE', 'HV211': 'MOTORCYCLE', 'HV212': 'CAR',
                                              'HV213': 'FLOOR',
                                              'HV216': 'BEDROOMS', 'HV221': 'PHONE', 'HV243A': 'CELLPHONE'})

        # recode water source, floor type and toilet type to score (1-5)
        # according to sustainlab groups recode dict
        survey_df = survey_df.replace({'FLOOR': self.floor_recode,
                                       'WATER_SOURCE': self.water_recode,
                                       'TOILET_TYPE': self.toilet_recode
                                       })
        if survey_df.PHONE.dropna().empty:
            survey_df['PHONE'] = survey_df.CELLPHONE
        # recode urban/rural to binary
        survey_df['URBAN_RURAL'] = survey_df.URBAN_RURAL.replace({1: 0, 2: 1})
        # calculate the rooms per person from number of bedrooms and number of household members
        survey_df['ROOMSPP'] = survey_df.BEDROOMS / survey_df.MEMBERS

        # remove rows (households) that contain missing values
        survey_df = survey_df.dropna()

        # get year and country for filename
        year = survey_df.YEAR.apply(int).max()
        country = self.country_code_dict[filename[:2]]

        x = pd.DataFrame(survey_df.loc[:, self.features].values)
        y = survey_df.loc[:, self.info].values

        # standardize values
        x = StandardScaler().fit_transform(x)

        # print(f'{country} {year}')
        # print(f'Entries: {x.shape}')
        # print(f'Mean equal: {np.mean(x)}')
        # print(f'Standard deviation: {np.std(x)}', end='\n\n')

        # perform PCA
        pca = PCA(n_components=1)

        # create new DataFrame including asset wealth
        wealth_df = pd.DataFrame(data=pca.fit_transform(x), columns=['WEALTH_INDEX'])

        # add cluster number, urban/rural info and survey year

        wealth_df = pd.concat([wealth_df, survey_df[['CLUSTER', 'URBAN_RURAL', 'YEAR']]], axis=1)

        # get number of households per cluster
        n = wealth_df.CLUSTER.value_counts(sort=False).to_list()
        # calculate mean values for asset wealth, urban/rural and survey year per cluster
        wealth_df = wealth_df.groupby('CLUSTER').mean()[['WEALTH_INDEX', 'URBAN_RURAL', 'YEAR']]

        # add additional columns and remove duplicates
        wealth_df['COUNTRY'] = country
        wealth_df['COUNTRY_CODE'] = filename[:2]
        wealth_df = wealth_df.merge(survey_df[['HV000', 'CLUSTER']].drop_duplicates(), on='CLUSTER')
        wealth_df['YEAR'] = wealth_df.YEAR.apply(int)
        wealth_df = wealth_df.rename(columns={'YEAR':'SURVEY_YEAR'})
        wealth_df = wealth_df.reset_index()
        wealth_df['n'] = n

        wealth_df.to_csv(os.path.join(self.wealth_path, country + '_' + str(year) + '.csv'), index=False)

    def create_wealth_geo_df(self, shape_file: str):
        '''Combine survey data including Asset Wealth with geocoordinates.

        Args:
            shape_file: Filename of DHS shapefile


        '''

        # load shapefile to geopandas DataFrame
        dhs_geo_df = gpd.read_file(os.path.join(self.shape_path, shape_file))

        # select required columns
        dhs_geo_df = dhs_geo_df[['DHSCC', 'DHSYEAR', 'DHSCLUST', 'URBAN_RURA', 'LATNUM', 'LONGNUM']]

        # recode urban/rural to binary
        dhs_geo_df['URBAN_RURA'] = dhs_geo_df['URBAN_RURA'].replace({'U': 0, 'R': 1})
        dhs_geo_df['DHSYEAR'] = dhs_geo_df.DHSYEAR.apply(int)

        # get country to load wealth_df
        country = self.country_code_dict[dhs_geo_df['DHSCC'].iloc[0]]
        wealth_file = [string for string in os.listdir(self.wealth_path) if country in string][0]
        wealth_df = pd.read_csv(os.path.join(self.wealth_path, wealth_file))

        # merge asset wealth and geo dataframe
        dhs_geo_df = dhs_geo_df.merge(wealth_df, left_on=['DHSCLUST'], right_on=['CLUSTER'])[[
            'DHSCC', 'DHSYEAR', 'DHSCLUST', 'HV000', 'URBAN_RURA', 'LATNUM', 'LONGNUM', 'WEALTH_INDEX', 'SURVEY_YEAR',
            'COUNTRY', 'n']]

        # remove NaN values
        dhs_geo_df.dropna()
        # concat_df = concat_df.append(dhs_geo_df)
        # write DataFrame to file
        dhs_geo_df.to_csv(os.path.join(self.geo_wealth_path, wealth_file), index=False)

    def split_sustainlab_clusters(self):
        '''Split sustainlab cluster csv into separate csv files. Creates one csv file per survey (country/year).
        '''
        try:
            dhs_sustainlab_df = pd.read_csv(os.path.join(self.geo_wealth_path, self.sustainlab_group_file))
            # remove NaN values
            dhs_sustainlab_df = dhs_sustainlab_df.dropna()
            # rename to match new DataFrames
            dhs_sustainlab_df = dhs_sustainlab_df.rename(columns={'cluster': 'DHSCLUST',
                                                                  'svyid': 'DHSYEAR',
                                                                  'wealth': 'WEALTH_INDEX',
                                                                  'cname': 'DHSCC',
                                                                  'hv000': 'HV000',
                                                                  'year': 'SURVEY_YEAR',
                                                                  'country': 'COUNTRY',
                                                                  'households': 'n',
                                                                  })[["DHSCLUST", "DHSYEAR", "WEALTH_INDEX", "HV000",
                                                                      "SURVEY_YEAR", "DHSCC", "COUNTRY", "n",
                                                                      "LATNUM", "LONGNUM", "URBAN_RURA"]]

            dhs_sustainlab_df['DHSYEAR'] = pd.to_numeric(dhs_sustainlab_df.DHSYEAR.apply(lambda x: x[-4:]),
                                                         downcast ='integer',errors='coerce')
            # recode urban/rural to binary
            dhs_sustainlab_df['URBAN_RURA'] = dhs_sustainlab_df.URBAN_RURA.replace({'U': 0, 'R': 1})

            # remove all clusters that are older than 2012

            dhs_sustainlab_df = dhs_sustainlab_df[dhs_sustainlab_df.DHSYEAR >= 2012]
            dhs_sustainlab_df.dropna()
            grouped = dhs_sustainlab_df.groupby(['DHSYEAR', 'COUNTRY'])
            # write clusters to separate files
            for name, group in grouped:
                print(name[1] + str(name[0]))
                group.to_csv(os.path.join('./geo_wealth_dhs/', name[1] + str(int(name[0])) + '.csv'))
        except FileNotFoundError:
            print('Sustainlab Group Clusters not found')


def main():
    '''Recode dhs survey data and create label csv file for each survey including Asset Wealth and geocoordinates.
    '''
    #### DEFINE VARIABLES
    floor_recode_df, meta = pyreadstat.read_dta('./floor_recode.dta')
    toilet_recode_df, meta = pyreadstat.read_dta('./toilet_recode.dta')
    water_recode_df, meta = pyreadstat.read_dta('./water_recode.dta')

    floor_recode_dict = floor_recode_df.set_index('floor_code').to_dict()['floor_qual']
    toilet_recode_dict = toilet_recode_df.set_index('toilet_code').to_dict()['toilet_qual']
    water_recode_dict = water_recode_df.set_index('water_code').to_dict()['water_qual']
    water_recode_dict[63] = 4

    country_codes = requests.get(
        'http://api.dhsprogram.com/rest/dhs/countries?returnFields=CountryName,DHS_CountryCode').text
    country_code_dict = {}
    for country in json.loads(country_codes)['Data']:
        country_code_dict[country['DHS_CountryCode']] = country['CountryName'].replace(' ', '_')

    features = ['WATER_SOURCE', 'TOILET_TYPE', 'ELECTRICITY', 'RADIO', 'TV',
                'FRIDGE', 'MOTORCYCLE', 'CAR', 'FLOOR', 'ROOMSPP', 'PHONE', 'CELLPHONE']  # ,'cellphone']

    info = ['HHID', 'YEAR']
    dhs_survey_path = '/mnt/datadisk/data/surveys/asset/dhs_data/raw_data/household_data/'
    wealth_path = '/mnt/datadisk/data/surveys/asset/dhs_data/label_data/'
    shape_path = '/mnt/datadisk/data/surveys/asset/dhs_data/raw_data/geo_data/'
    dhs_surveys = [file for directory in os.listdir('./dhs_household_data/') for file in directory if file.endswith('.SAS7BDAT')]
    geo_wealth_path = '/mnt/datadisk/data/surveys/asset/dhs_data/label_data/'
    sustainlab_group_file = '/mnt/datadisk/data/surveys/asset/dhs_data/dhs_clusters_sustainlab_group.csv'

    # initiate Class Object
    dhs_obj = DHS_preparation(floor_recode=floor_recode_dict,
                              toilet_recode=toilet_recode_dict,
                              water_recode=water_recode_dict,
                              country_code_dict=country_code_dict,
                              features=features,
                              info=info,
                              dhs_survey_path=dhs_survey_path,
                              wealth_path=wealth_path,
                              shape_path=shape_path,
                              geo_wealth_path=geo_wealth_path,
                              sustainlab_group_file=sustainlab_group_file)
    # perform dhs preparation
    # calculate wealth index and create csv files per country_year
    for filename in os.listdir(dhs_obj.dhs_survey_path):
        dhs_obj.recode_and_format_dhs(filename)
    # add geo data and create new files
    for shapefile in os.listdir(shape_path):
        dhs_obj.create_wealth_geo_df(shapefile)
    # split sustainlab clusters to files per country_year
    dhs_obj.split_sustainlab_clusters()


if __name__ == "__main__":
    main()
