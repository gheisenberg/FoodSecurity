import os
import pyreadstat
# os.environ["MODIN_ENGINE"] = "ray"  # Modin will use dask/ray
# import modin.pandas as pd
import pandas as pd
import dask.dataframe as dd
from datetime import datetime
from collections import Counter
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from collections import defaultdict, OrderedDict
from icecream import ic
import numpy as np
import re
import ast
from warnings import warn
from openai import OpenAI
from time import sleep
from tqdm import tqdm
from openai import OpenAI
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import functools
from memory_profiler import profile
#import sleep
from time import sleep
from ethiopian_date import EthiopianDateConverter
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/water_sven/'
print(parent_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import helper_utils as hu
import visualizations as vis
import geo_utils as gu



ic_file = 'icecream_preprocessing.log'
def output_to_file(*args):
    with open(ic_file, 'a') as f:
        print(*args, file=f)

if os.path.exists(ic_file):
    os.remove(ic_file)
ic.configureOutput(outputFunction=output_to_file)



def initialize_sqlite_cache(db_file):
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value BLOB
            )
        """)
        conn.commit()

# Initialize some global variables
global_pattern = re.compile(r'[!\"#$%&\'()*+\-./:;<=>?@[\\\]^_`{|}~,]')
global_pattern_special = re.compile(r'["\'\\]')
global_pattern_folders = re.compile(r'[^\w\-\.]')  


def read_sav_with_fallback(file, *args, **kwargs):  
    if 'usecols' in kwargs:
        if kwargs['usecols'] is False:
            kwargs['usecols'] = None
    
    try:
        _, meta = pyreadstat.read_sav(file, metadataonly=True)
    except pyreadstat.ReadstatError as e:
        ic(f"ReadstatError: {e} when reading {file}. Skipping.")
        return None, None
    if 'metadataonly' in kwargs and kwargs['metadataonly']:
        # ic('metadataonly')
        return _, meta

    try:
        #ic(meta.file_encoding)
        df, meta = pyreadstat.read_sav(file, *args, encoding=meta.file_encoding, **kwargs)
        return df, meta

    except pyreadstat.ReadstatError as e:
        # pass
        ic(f"ReadstatError: {e} when reading {file} with encoding from meta {meta.file_encoding}. Skipping.")
        return None, None
    
def read_sav_with_fallback_old(file, *args, **kwargs):  
    if 'usecols' in kwargs:
        if kwargs['usecols'] is False:
            kwargs['usecols'] = None
    
    encodings = ['LATIN1', 'UTF-8']
    for encoding in encodings:
        try:
            df, meta = pyreadstat.read_sav(file, *args, encoding=encoding, **kwargs)
            # if 'usecols' in kwargs and kwargs['usecols'] is not None:
            #     missing_cols = set(kwargs['usecols']) - set(df.columns)
            #     ic(missing_cols)
            #     for col in missing_cols:
            #         if '_Nr' in col:
            #             nr = col.split('_Nr')[1][-1]
            #             try: 
            #                 nr = int(nr)
            #                 continue
            #             except ValueError:
            #                 pass    
            #         df[col] = pd.NA
            return df, meta
        except pyreadstat.ReadstatError as e:
            # pass
            ic(f"ReadstatError: {e} when reading {file} with encoding {encoding}. Trying next encoding.")

def load_single_file(file, cols=None, row_limit=False):
    """
    Load single .sav file.

    Args:
        file (str): Path to the .sav file.

    Returns:
        DataFrame: DataFrame containing data.
        meta: Meta data.
    """
    df, meta = read_sav_with_fallback(file, usecols=cols, row_limit=row_limit)
    return df, meta


def get_common_columns_new(files, verbose=False):
    results = []
    current_time = datetime.now()
    
    read_sav_with_limit = partial(read_sav_with_fallback, metadataonly=True)

    with ProcessPoolExecutor() as executor:
        # Create a mapping of futures to file names
        future_to_file = {executor.submit(read_sav_with_limit, file): file for file in files}
        
        for future in as_completed(future_to_file):
            # Get the file name from the future_to_file mapping
            file = future_to_file[future]
            try:
                result = future.result()
                if result[0] is not None:
                    results.append(result)
            except Exception as e:
                ic(e, "Error processing file:", file)
    if verbose >= 2:
        ic('time loading', datetime.now() - current_time)
        current_time2 = datetime.now()
        ic('results', len(results))
    
    
    def process_result(result):
        df, meta = result
        column_dict = {}
        for column_code, column_label in meta.column_names_to_labels.items():
            try:
                column_dict[column_code] = column_label.lower()
            except AttributeError as e:
                if column_code == 'HHID' and not 'household number' in meta.column_names_to_labels.values():
                    column_label = 'household number'
                else:
                    column_label = column_code
                column_dict[column_code] = column_label.lower()
                ic('decode data: AttributeError (col_code, col_label (replacement), file)', column_code, column_label, file)  # Replaces print
                ic(e)
                
        return {
            'columns': list(column_dict.values()),
            'column_names': set(column_dict.keys()),
            'label_to_names': {label: name for name, label in column_dict.items()}
        }

    
    with ThreadPoolExecutor() as executor:
        result_dicts = list(executor.map(process_result, results))

    # Create an ordered dictionary to collect unique columns and maintain order - Note this can/should? be a set?
    unique_columns_dict = OrderedDict()

    # Collect all sets of column names into a list
    all_column_names = [result_dict['column_names'] for result_dict in result_dicts]
    # ic('all_column_names', all_column_names)
    all_column_names = functools.reduce(set.union, all_column_names)
    # all_column_names = [set(c) for c in all_column_names]
    # Update label_to_names and unique_columns_dict
    label_to_names = defaultdict(set)
    for result_dict in result_dicts:
        for label, name in result_dict['label_to_names'].items():
            label_to_names[label].add(name)
        for col in result_dict['columns']:
            # Only add column to unique_columns_dict if it's not already present
            unique_columns_dict.setdefault(col, None)
    
    # Create a list of unique columns from the keys of unique_columns_dict
    all_column_labels = list(unique_columns_dict.keys())

    # Compute the intersection of column names to find common columns
    common_column_labels = set(result_dicts[0]['columns'])
    common_column_names = result_dicts[0]['column_names']
    label_to_names = defaultdict(set)
    for label, name in result_dicts[0]['label_to_names'].items():
        label_to_names[label].add(name)

    # Compute intersections and update label_to_names
    for result_dict in result_dicts[1:]:
        common_column_labels.intersection_update(set(result_dict['columns']))
        common_column_names.intersection_update(result_dict['column_names'])

    # Order the common columns according to their appearance in unique_columns_list
    common_column_labels = [col for col in all_column_labels if col in common_column_labels]
    common_column_names = [col for col in all_column_labels if col in common_column_names]

    if verbose >= 2:
        ic('time creating (un)common names/labels', datetime.now() - current_time2)
        current_time2 = datetime.now()
        ic('finished label and name intersection')

    # Uncommon labels and columns
    uncommon_column_names = set(all_column_names) - set(common_column_names)
    uncommon_column_labels = set(all_column_labels) - set(common_column_labels)

    # Create a names_to_labels dictionary
    names_to_labels = {name: label for label, names_set in label_to_names.items() for name in names_set}
    # Identify the missing names based on non-overlapping labels
    missing_names = [name for name in common_column_labels if names_to_labels.get(name) not in common_column_names]
    missing_label_details = defaultdict(set)
    for name in missing_names:
        for label, names in label_to_names.items():
            if name in names:
                missing_label_details[name].add(label)
    
    if verbose:
        ic('time creating missing labels', datetime.now() - current_time2)
        ic('overall time', datetime.now() - current_time)

    return list(common_column_labels), list(common_column_names), missing_label_details, uncommon_column_names, uncommon_column_labels, all_column_names, all_column_labels



###@cache_to_sqlite('my_cache.db')
def convert_non_numeric_to_str(col):
    temp_col = pd.to_numeric(col, errors='coerce')
    if temp_col.isna().all():  # No numeric values present
        return col.astype(str)
    return col  # Return original column if numeric values are present


def convert_case_id(row):
    # ic(row)
    case_id_str = row['case identification']
    case_id_str = case_id_str.lstrip()
    if pd.isna(row['cluster number']):
        if len(case_id_str) == 8:
            cluster_nr, hh_nr_str = case_id_str[:4], case_id_str[4:]
            # ic('0', case_id_str, cluster_nr, hh_nr_str)
        else:
            ic('0a', case_id_str)
            raise NotImplementedError('hh_nr_str has unexpected format: Not implemented for non 8 length case_id_strs')
    else:
        cluster_nr = str(int(row['cluster number']))
        if cluster_nr in case_id_str:
            hh_nr_str = case_id_str.split(cluster_nr, 1)[-1]
            # ic('1', hh_nr_str)
        else:
            #some cluster numbers not directly present in case id
            cnr = cluster_nr[-2:]
            cnr2 = ' ' + cluster_nr[-1:]
            case_id_str2 = case_id_str[2:]
            if cnr in case_id_str2:
                #should not be at the very beginning
                hh_nr_str = case_id_str2.split(cnr, 1)[-1]
                # ic(cnr, hh_nr_str)
            elif cnr2 in case_id_str2:
                hh_nr_str = case_id_str2.split(cnr2, 1)[-1]
                # ic(cnr2, hh_nr_str)
            else:
                warn(f'Caution!: Cluster nr not in case id {case_id_str, cluster_nr}')
            # ic('Cluster Nr not in case id', case_id_str, cluster_nr, hh_nr_str)
    if ' ' in hh_nr_str:
        hh_nr_str = hh_nr_str.replace(' ', '0')
        # ic('2', int(hh_nr_str))
        try:
            hh_nr_str = int(hh_nr_str)
        except ValueError:
            raise ValueError(f'hh_nr_str has unexpected format {hh_nr_str, case_id_str, cluster_nr}')
    else:
        try:
            hh_nr_str = int(hh_nr_str)
        except ValueError:
            raise ValueError('3', hh_nr_str, case_id_str, cluster_nr)
    if hh_nr_str == '':
        ic('empty hh nr', case_id_str, cluster_nr, hh_nr_str)
    return int(cluster_nr), int(hh_nr_str)


def convert_ethiopian_date(row):
    # Convert Ethiopian date to Gregorian date
    try:
        gregorian_date = EthiopianDateConverter.to_gregorian(row['year of interview'], row['month of interview'], row['day of interview'])
    except Exception as e:
        ic(row['year of interview'], row['month of interview'], row['day of interview'])
        ic(gregorian_date, e)
        return [None, None, None]
    year, month, day = gregorian_date.year, gregorian_date.month, gregorian_date.day
    return year, month, day
    

def decode_data(file, folder_path, cols=None, row_limit=False, col_labels=False, create_GEID=False, create_DHSID=False, create_version_nr=False, load_keys=True,
                try_loading_nr_columns=True, load_gregorian_calender=True):
    """
    Decode .sav file data.

    Args:
        file (str): Path to the .sav file.

    Returns:
        DataFrame: Decoded DataFrame.
        meta: Meta data.
    """
    cols_to_drop = []
    av_l = []

    if col_labels:
        if 'country code and phase' not in col_labels:
            # cols_to_drop.append('country code and phase')
            col_labels.append('country code and phase')
        if create_DHSID or create_GEID or create_version_nr:
            for c in ['cluster number', 'household number', 'case identification']:
                if c not in col_labels:        
                    col_labels.append(c)
                    cols_to_drop.append(c)
        if try_loading_nr_columns:
            for i in range(1, 10):
                for col in col_labels.copy():
                    col_labels.append(f"{col}_Nr{i}")
        
        #ensure all date stuff is loaded, if date stuff is loaded
        for date_str in ['day', 'month', 'year']:
            if date_str + ' of interview' in col_labels:
                av_l.append(date_str + ' of interview')
        if len(av_l) != 0 and len(av_l) != 3:        
            for date_str in ['day', 'month', 'year']:
                if date_str + ' of interview' not in col_labels:
                    col_labels.append(date_str + ' of interview')
            if not 'country code and phase' in col_labels:
                col_labels.append('country code and phase')
                    
        cols = get_column_names(file, labels=col_labels)
        if not cols:
            cols = ['country code and phase']
    
    df, meta = load_single_file(file, cols=cols, row_limit=row_limit)

    if df is None:
        return None, None
    # Decode column names and values
    column_dict = {}
    variable_dict = {}
    for column_code, column_label in meta.column_names_to_labels.items():
        try:
            column_dict[column_code] = column_label.lower()
        except AttributeError as e:
            if column_code == 'HHID' and not 'household number' in meta.column_names_to_labels.values():
                column_label = 'household number'
            else:
                column_label = column_code
            column_dict[column_code] = column_label.lower()
            ic('decode data: AttributeError (col_code, col_label (replacement), file)', column_code, column_label, file)  # Replaces print
            ic(e)
        if column_code in meta.variable_value_labels.keys():
            # ic('hi1')
            if load_keys:
                variable_dict[column_label.lower()] = {k: f"{k}: {v.lower()}" for k, v in meta.variable_value_labels[column_code].items()}
            else:
                variable_dict[column_label.lower()] = {k: (v.lower() if v else k) for k, v in meta.variable_value_labels[column_code].items()}
                
            # one single survey with 'country code and phase' as a variable
            if 'country code and phase' in variable_dict:
                del variable_dict['country code and phase']
            
                
    df = df.rename(columns=column_dict)
    if len(df) > 0:
        df = df.replace(variable_dict)
        # Conver non-numeric columns and columns without ints/floats in them to string type
        df = df.apply(convert_non_numeric_to_str, axis=0)
        # Replace 'nan' strings with np.NaN
        df = df.replace({'nan': np.NaN, '': np.NaN})#, '0.0: ': np.NaN, '1.0: ': np.NaN})
        # Drop columns with all NaN values
        #df = df.dropna(axis=1, how='all')

    folder = os.path.dirname(file)
    folder = os.path.basename(folder)

    ###Create IDs
    for c in ['cluster number', 'household number']:
        if c in df.columns:
            # Attempt to convert the column to 'Int32', coercing errors to NaN
            df_temp = df[c].astype('float').astype('Int32')

            # Create a mask where each value is True if the corresponding value in the original column is not a numeric string, and False otherwise
            conversion_failed = df[c].notna() & pd.to_numeric(df[c], errors='coerce').isna()
            # Print the rows where the conversion failed
            if len(df.loc[conversion_failed, c]) > 0:
                ic(df.loc[conversion_failed, c], file)
                warn(f'Conversion failed for {c} in {file}')
            df[c] = df_temp
    
    if 'cluster number' in df.columns and 'household number' in df.columns:
        ###Replace non unique household numbers
        dup = df[df.duplicated(subset=['cluster number', 'household number'], keep=False)]
        if len(dup) > 0:
            # ic(dup[['cluster number', 'household number', 'case identification']].value_counts())
            # ic(df['case identification'].value_counts())
            if df['case identification'].value_counts().max() > 1:
                ic(df[['cluster number', 'household number', 'case identification']].sample(50))
                raise ValueError('non unique cluster + hh nr and non unique case identification - skipping survey')
            else:
                # ic('in', df[['cluster number', 'case identification', 'household number']].sample(20))
                new_columns = pd.DataFrame(df[['cluster number', 'case identification']].apply(convert_case_id, axis=1).tolist(), 
                            columns=['cluster number from case id', 'household number from case id'], index=df.index)
                # ic('out')
                df['household number'] = new_columns['household number from case id']
                df['generated hhnr from case id'] = True
                # replace NaNs in cluster number with cluster number from case id
                if pd.isna(df['cluster number']).any():
                    # ic('NaNs in cluster number', pd.isna(df['cluster number']).sum())
                    df.at[df['cluster number'].isna(), 'generated clnr from case id'] = True
                    df['cluster number'] = df['cluster number'].fillna(new_columns['cluster number from case id'])

                # ic(df[['cluster number', 'case identification', 'household number']].sample(20))
                # warn(f'Reconstructed houhehold number from case id {folder}')
            
    if create_GEID or create_DHSID:
        # if not cols or 'cluster number' in cols or 'household number' in cols:
        if create_GEID:
            df['GEID'] = np.NaN
        if create_DHSID:
            df['DHSID'] = np.NaN
        
                        
        if create_GEID:
            GEID_init = retrieve_gps_file(file, folder_path)
            if GEID_init:
                df['GEID'] = GEID_init + df['cluster number'].astype(str).str.zfill(8)
                
        if create_DHSID:
            mask = df['cluster number'].notna()
            # Use the mask to update the 'DHSID' column only where the mask is True
            df.loc[mask, 'DHSID'] = folder + df.loc[mask, 'cluster number'].astype(str).str.zfill(8)
            # Check for duplicates
            dup = df[df.duplicated(subset=['DHSID', 'household number'], keep=False)]
            if len(dup) > 0:
                ic(dup)
                raise ValueError(f'duplicates in DHSID + hhnr {folder}')
            mask = df['DHSID'].notna() & df['household number'].notna()
            df.loc[mask, 'DHSID + HHID'] = df.loc[mask, 'DHSID'] + \
                df.loc[mask, 'household number'].astype(str).str.zfill(8)# Apply the mask to the DataFrame
            #show duplicates
            dup = df[df.duplicated(subset=['DHSID + HHID'], keep=False)]
            if len(dup) > 0:
                raise ValueError(f'duplicates in DHSID + HHID {folder}\n\n')
    
    if create_version_nr:
        if not folder:
            folder = os.path.dirname(file)
            folder = os.path.basename(folder)
        
        try:
            df['version_nr'] = df['country code and phase'].str[2].astype(int)
        except ValueError:
            printed = False
            for i, v in enumerate(df['country code and phase']):
                try:
                    df['version_nr'][i] = int(v[2])
                except:
                    if not printed:
                        ic(v)
                        warn('country code and phase error')
                    printed = True
                    #raise ValueError('IndexError')
            df['version_nr'] = int(folder[4])
        df['subversion_nr'] = folder[5]
    
    if av_l:
        if 'ET' in next(iter(df['country code and phase'])):
            for date_col in ['year of interview', 'month of interview', 'day of interview']:
                ic(date_col, df[date_col].dtype, df[date_col].head())
                df[date_col] = df[date_col].astype('Int32')
            ic(df[['year of interview', 'month of interview', 'day of interview']].head())
            df[['year of interview', 'month of interview', 'day of interview']] = df.apply(convert_ethiopian_date, axis=1, result_type='expand')
            # df[['year of interview', 'month of interview', 'day of interview']] = df[['year of interview', 'month of interview', 'day of interview']].apply(convert_ethiopian_date, axis=1)
            ic('converted ethiopian date', df[['year of interview', 'month of interview', 'day of interview']].head())
            
        
    if cols_to_drop:
        cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(cols_to_drop, axis=1)
    return df, meta

def get_column_names(file, labels=None):
    _, meta = read_sav_with_fallback(file, metadataonly=True)
    if meta is None:
        return None
    if labels is not None:
        # Create a mapping from labels to names
        label_to_name = {v.lower() if v else k.lower(): k for k, v in meta.column_names_to_labels.items()}
        # Map the specified labels to names
        # print(labels)
        col_names = [label_to_name[label.lower()] for label in labels if label.lower() in label_to_name]
        return col_names
    else:
        return list(meta.column_names)

def load_data(files, dataset_type, folder_path, cols=False, col_labels=None, create_GEID=False, create_DHSID=False, 
              max_workers=False, create_version_nr=False, verbose=False, load_keys=True):
    """
    Load and decode all relevant .sav files in a folder.

    Args:
        folder_path (str): Path to the root folder containing dataset folders.
        dataset_type (str): Type of the dataset to load.

    Returns:
        dict: Dictionary containing all loaded DataFrames.
        dict: Dictionary containing all meta data.
    """
    partial_decode_data = partial(decode_data, folder_path=folder_path, cols=cols, col_labels=col_labels, create_GEID=create_GEID,
                                  create_DHSID=create_DHSID, create_version_nr=create_version_nr, load_keys=load_keys)
    # Using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(partial_decode_data, files))  
        
    # Separating Dask DataFrames and metadata
    dfs, metas = zip(*results)
    dfs = [df for df in dfs if df is not None]
    metas = [meta for meta in metas if meta is not None]
    
    all_data = {dataset_type: dfs}
    meta_data = {dataset_type: metas}
    if verbose:
        ic('found', len(all_data[dataset_type]), dataset_type)  # Replaces print
    return all_data, meta_data


def get_dhs_files(folder_path, dataset_type, min_version, excl_countries=[], max_version=False):
    files = []
    for folder in os.listdir(folder_path):
        #ic(folder)
        if int(folder[4]) < min_version:
            continue
        elif max_version and int(folder[4]) > max_version:
            continue
        if folder[2:4] == dataset_type:
            if folder[0:2] in excl_countries:
                ic('dropping', folder)
                continue
            folder_full_path = os.path.join(folder_path, folder)
            for file in os.listdir(folder_full_path):
                if file.endswith('.sav') or file.endswith('.SAV'):
                    files.append(folder_full_path + '/' + file)
    files = sorted(files)
    return files

def get_files_by_extension(folder_path, ext='.csv'):
    files = []
    for f in os.listdir(folder_path):
        if f.endswith(ext) or f.endswith(ext.upper()):
            files.append(folder_path + '/' + f)
    files = sorted(files)
    return files


def combine_survey_dfs_new(all_data, drop_col_counter=False, verbose=False, meta_df=False):
    """
    Explore and combine DataFrames stored in all_data.

    Args:
        all_data (dict): Dictionary containing dataset_type as keys and lists of DataFrames as values.
        drop_col_counter (int): Drop column duplicates with more than drop_col_counter.

    Returns:
        list: List of combined DataFrames.
    """
    combined_dfs = []
    for dataset_type, ddf in all_data.items():
        #ic(f"Dataset Type: {dataset_type}")
        ddf = list(ddf)
        dfs_list = []
        for i, df in enumerate(ddf):
            new_columns = []
            duplicate_counter = Counter()  # To keep track of duplicate columns within the same DataFrame
            drop_cols = []  # Initialize an empty list for columns to drop for each DataFrame
    
            # Rename duplicate columns in the same DataFrame
            for col in df.columns:                
                duplicate_counter[col] += 1            
                if duplicate_counter[col] > 1:  
                    new_col = f"{col}_Nr{duplicate_counter[col]}"
                else:
                    new_col = col
                    
                new_columns.append(new_col)    
                # Check if the column should be dropped
                if drop_col_counter and duplicate_counter[col] > drop_col_counter:
                    drop_cols.append(new_col)
            df.columns = new_columns
            # Drop columns if needed
            if drop_cols:
                df = df.drop(drop_cols, axis=1)
            #ic(new_columns)
            #ic(df[new_columns])
            dfs_list.append(df)
            if not 'DHSID' in df.columns:
                ic('DHSID missing', df.columns)
                raise ValueError('DHSID missing, necessary now, change create_DHSID = True')
        
        if verbose:
            ic('Combining DataFrames...')
        current_time = datetime.now()
        
        # join dfs on DHSID
        if meta_df is not False:
            combined_df = meta_df
        else:
            combined_df = dfs_list[0]
            dfs_list.remove(combined_df)
            
        cols = set(combined_df.columns)
        ic(combined_df)
        for df in dfs_list:
            # ic(cols)
            # ic(df.columns)
            dfcols = list(df.columns)
            cols_to_combine = list(set(dfcols) - cols) + ['DHSID']
            df = df[cols_to_combine]
            ic(cols_to_combine)
            ic(df, '\n')
            combined_df = combined_df.merge(df[cols_to_combine], on='DHSID', how='outer')

        # combined_df = pd.concat(ddf, join='outer', axis=0)

        # # Concatenate Pandas DataFrames
        # combined_df = pd.concat(ddf, join='outer', axis=0)
        combined_df = combined_df.apply(convert_non_numeric_to_str, axis=0)
        for col in combined_df.columns:
            if combined_df[col].dtype == 'object':
                #check for numeric types
                combined_df[col] = combined_df[col].astype(str)
            #if col != 'version_nr' and col != 'subversion_nr' and ('_nr' in col or '_NR' in col or '_Nr' in col):
             #   warn(f'Nr in col? {col}')
        combined_df = combined_df.replace({'nan': np.NaN, '': np.NaN})#, '0.0: ': np.NaN, '1.0: ': np.NaN})
        combined_df = combined_df.reset_index(drop=True)
        if verbose:
            ic('concat time', datetime.now() - current_time)
        combined_dfs.append(combined_df)
    return combined_dfs
    
    
def combine_survey_dfs(all_data, drop_col_counter=False, verbose=False):
    """
    Explore and combine DataFrames stored in all_data.

    Args:
        all_data (dict): Dictionary containing dataset_type as keys and lists of DataFrames as values.
        drop_col_counter (int): Drop column duplicates with more than drop_col_counter.

    Returns:
        list: List of combined DataFrames.
    """
    combined_dfs = []
    for dataset_type, ddf in all_data.items():
        #ic(f"Dataset Type: {dataset_type}")
        ddf = list(ddf)
        unique_ids_s = set()
        for i, df in enumerate(ddf):
            new_columns = []
            duplicate_counter = Counter()  # To keep track of duplicate columns within the same DataFrame
            drop_cols = []  # Initialize an empty list for columns to drop for each DataFrame
    
            # Rename duplicate columns in the same DataFrame
            for col in df.columns:                
                duplicate_counter[col] += 1            
                if duplicate_counter[col] > 1:  
                    new_col = f"{col}_Nr{duplicate_counter[col]}"
                else:
                    new_col = col
                    
                new_columns.append(new_col)    
                # Check if the column should be dropped
                if drop_col_counter and duplicate_counter[col] > drop_col_counter:
                    drop_cols.append(new_col)
            df.columns = new_columns
            # Drop columns if needed
            if drop_cols:
                df = df.drop(drop_cols, axis=1)
            #ic(new_columns)
            #ic(df[new_columns])
            if 'DHSID + HHID' in df.columns:
                unique_ids_sl = set(df['DHSID + HHID'])
                # if len(unique_ids_sl) < len(df):
                #     ic('duplicated in df', df[df.duplicated(subset=['DHSID + HHID'], keep=False)])
                
                #test for overlap in existing ids
                if len(unique_ids_s.intersection(unique_ids_sl)) > 0:
                    ic('duplicates with other dfs', unique_ids_s.intersection(unique_ids_sl))
                
                unique_ids_s = unique_ids_s.union(unique_ids_sl)
                
            ddf[i] = df
        
        if verbose:
            ic('Combining DataFrames...')
        current_time = datetime.now()
        

        # Concatenate Pandas DataFrames
        combined_df = pd.concat(ddf, join='outer', axis=0)
        combined_df = combined_df.apply(convert_non_numeric_to_str, axis=0)
        for col in combined_df.columns:
            if combined_df[col].dtype == 'object':
                #check for numeric types
                combined_df[col] = combined_df[col].astype(str)
            #if col != 'version_nr' and col != 'subversion_nr' and ('_nr' in col or '_NR' in col or '_Nr' in col):
             #   warn(f'Nr in col? {col}')
        combined_df = combined_df.replace({'nan': np.NaN, '': np.NaN})#, '0.0: ': np.NaN, '1.0: ': np.NaN})
        combined_df = combined_df.reset_index(drop=True)
        if verbose:
            ic('concat time', datetime.now() - current_time)
        combined_dfs.append(combined_df)
    return combined_dfs
    

def retrieve_gps_file(file_path, folder_path):
    """
    Retrieve the GPS file for a given .sav file.

    Args:
        file_path (str): Path to the .sav file.

    Returns:
        str: Path to the GPS file.
    """
    # Get the folder name
    folder = os.path.dirname(file_path)
    folder = os.path.basename(folder)
    country = folder[:2]
    version_nr = folder[4]
    version_subnr = folder[5]
    try:
        version_subnr = int(version_subnr)
    except ValueError:
        pass
    match_str = f"{country}GE{version_nr}"
    match_str = f"{country}GE{version_nr}"
    #match in path
    matched_folders = []
    matched = False
    sub_version_def_l = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], ['I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q'], ['R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']]
    for root, dirs, files in os.walk(folder_path):
        break
    for dir in dirs:
        if match_str in dir:
            for sub_v in sub_version_def_l:
                if version_subnr in sub_v:
                    dir_sub_v = dir[5]
                    try:
                        dir_sub_v = int(dir_sub_v)
                    except ValueError:
                        pass
                    if dir_sub_v in sub_v:
                        matched_folders.append(dir)
    new_matched = []
    if len(matched_folders) > 1:
        for mf in matched_folders:
            if len(mf) == 8:
                new_matched.append(mf)
            else:
                warn(f'Caution: found more than one folder {folder} matched: {matched_folders} dropping {mf}')
                
                
        if len(new_matched) == 1:
            matched = new_matched[0]
        elif not new_matched:
            raise ValueError('Caution1: found no folder! Matched but wrong sub version?:', folder, 'matched:', matched_folders)
        else:
            ic(new_matched)
            ic(len(nm) for nm in new_matched)
            raise ValueError('Caution: found more than one folder', folder, 'matched:', matched_folders)
    elif not matched_folders:
        pass
    else:
        matched = matched_folders[0]
    return matched

def test_str_for_numeric(answer, col_n):
    answer_out = False
    answer_in = answer
    # if split_str:
    #     answer = answer.split(split_str)
    # if use_right_split:
    #     id = -1
    # else:
    #     id = 0
    test_a = answer
    test_b = False
    # if len(answer) > 2:
    #     warn(f'"{split_str}": 2 or more in answer', answer, answer_in, id)
    if ' ' in test_a:
        test_l = test_a.split(' ')
        test_b = ''
        for t in test_l:
            if t == '' or t == ' ':
                continue
            if '+' in t:
                t = t.replace('+', '')
            if ',' in t:
                t = t.replace(',', '')
            try:
                test_a = float(t)
            except ValueError:
                if t == 'one':
                    test_a = 1
                elif t == 'two':
                    test_a = 2
                else:
                    test_b += t + ' '
        if test_b != '':
            test_b = test_b[:-1]
    try:
        answer_out = float(test_a)
    except ValueError:
        # if test_a == 'less than  minute':
        #     answer_out = 1
        # else:
        ic(f'Conversion failed {col_n}', answer_in, test_a, test_b)

    if (test_b and not test_b in col_n):
        km_s = set(['km', 'kilometer', 'kilometre', 'kms', 'kilometers', 'kilometres'])
        min_s = set(['minute', 'minutes', 'min'])
        if any([True for k in km_s if k in test_b]) and any([True for c in km_s if c in col_n]):
            pass
        elif any([True for k in min_s if k in test_b]) and any([True for c in min_s if c in col_n]):
            pass
        elif 'year' in test_b and 'month' in col_n:
            answer_out = answer_out * 12
        else:
            warn(f"test strs: {test_b} \n"
                f"Do not change: {answer_in} for {col_n} {id}")
            answer_out = False
    elif test_b and 'visits' in test_b:
        answer_out = False
    #ic(col_n, answer_in, answer_out, test_a, test_b)
    if test_b:
        ic(col_n, test_b, answer_in, answer_out)
    return answer_out

def auto_test_numeric(answer, row, answer_list_w_keys):
    answer_out = False
    numeric_percent = row['numeric [%]']
    if answer and pd.notna(numeric_percent) and numeric_percent != 0:
        if ' ' == answer[0]:
            answer = answer[1:]
        if ' ' == answer[-1]:
            answer = answer[:-1]
        try:
            answer_out = float(answer)
            return answer_out
        except ValueError:
            pass
        
        if ('none' in answer or 'noe' in answer or answer in {'no', 'non', 'not', 'none', 'noe'}) \
                and not any([True for a in answer_list_w_keys if 'yes' in a]):
            return 0
        found = False
        for r_str in ['+', 'more than ', 'less than ', ' and more', ' or more', ' and less', ' or less']:
            if r_str in answer:
                found = True
                answer = answer.replace(r_str, '')
        if found:
            answer_out = test_str_for_numeric(answer, row['column_name'])
            
            
        # elif '+' in answer:
        #     answer_out = test_str_for_numeric(answer, False, row['column_name'])
        # elif 'more' in answer:
        #     if 'more than ' in answer:
        #         answer_out = test_str_for_numeric(answer, 'more than ', row['column_name'], use_right_split=True)
        #     #elif ('unknown' == answer or "don't know" == answer or 'dk' == answer):
        #     #   answer_out = np.NaN

        #     elif ' or more' in answer:
        #         answer_out = test_str_for_numeric(answer, ' or more', row['column_name'])
            
        #     elif ' and more' in answer:
        #         answer_out = test_str_for_numeric(answer, ' and more', row['column_name'])
                
        # elif 'less than ' in answer:
        #     answer_out = test_str_for_numeric(answer, 'less than ', row['column_name'], use_right_split=True)    
    elif not answer:
        warn(f'No answer {row["column_name"]} {answer}')

    return answer_out

def auto_known_issues(answer, row):
    answer_out = False
    if answer == 'dk':
        answer_out = "don't know"
    #elif answer == 'noe':
     #   answer_out = 'none'
    elif answer == 'unknow' or answer == 'unknown':
        answer_out = "don't know"
    
    elif '..ue ' in answer:
        answer_out = answer.replace('..ue ', '')
    elif '..le ' in answer:
        answer_out = answer.replace('..le ', '')
    elif '..' in answer:
        answer_out = answer.replace('..', '')
    elif 'ÿû ' in answer:
        answer_out = answer.replace('ÿû ', '')
    elif 'ÿ' in answer:
        answer_out = answer.replace('ÿ', '')
    return answer_out

#@cache_to_sqlite('my_cache.db')    
def autocorrect(answer_list_w_keys, row):
    """
    Cases:
    1. key without answer
    2. int as key
    3. key as int"""
    kv_dict = defaultdict(list)
    kv_dict_corrected = {}
    kv_dict_corrected2 = {}
    for answer_n_key in answer_list_w_keys:
        answer = False           
        if ': ' in answer_n_key:
            key, answer = answer_n_key.split(': ', 1)
            #key without answer
            if not answer:
                #ic('no answer', answer_n_key, key, answer)
                key = answer_n_key.split(': ', 1)[0]
        else:
            #no key
            key = answer_n_key
        try:
            key = float(key)
        except ValueError:
            #no key
            #ic(row['column_name'], 'key not floatable', key, type(key), answer_n_key, answer)
            pass
        #if answer is False or not (isinstance(answer, float) or isinstance(answer, int)):
        kv_dict[key].append((answer_n_key, answer))
    
    for key, answers in kv_dict.items():
        #ic(key, answers)
        for (answer_n_key, answer) in answers:
            #key without answer?
            if not answer:
                if answers[0][1]:
                    kv_dict_corrected[str(key) + ': '] = answers[0][1]
                    kv_dict_corrected2[answer_n_key] = answers[0][1]
                    ic('no answer', row['column_name'], answer_n_key, key, answer, 'corrected to', answers[0][1])
                    continue
                else:
                    #ic('no answer not corrected', row['column_name'], answer_n_key, key, answer, 'not corrected', answers)
                    #ic('setting answer to key', key)
                    answer = key
            #solves key as int as well as "more than X" etc
            if isinstance(answer, str):
                answer_out = auto_test_numeric(answer, row, answer_list_w_keys)
            else:
                answer_out = answer
                
            if answer_out is not False:
                kv_dict_corrected[answer] = answer_out
                kv_dict_corrected2[answer_n_key] = answer_out  
            else:
                #maps 'unknown' and 'dk' to "don't know" and similar
                answer_out2 = auto_known_issues(answer, row)
                if answer_out2 is not False:
                    kv_dict_corrected[answer] = answer_out2 
                    kv_dict_corrected2[answer_n_key] = answer_out2               
                
    kv_dict_dubious = {k: v for k, v in kv_dict.items() if len(v) != 1}
    return kv_dict_dubious, kv_dict_corrected, kv_dict_corrected2, kv_dict


def create_unq_answers(top_values):
    d = {}
    for name, v in top_values.items():
        try:
            name2 = name.split(': ', 1)[1]
        except IndexError:
            name2 = name.split(': ', 1)[0]
            #ic(name, name2)
        name = name2
        if name not in d:
            d[name] = v
        else:
            d[name] += v

    # Create a new Series with the same name as top_values
    unq_answers = pd.Series(d, name=top_values.name)

    # Copy the index names from top_values
    if isinstance(top_values.index, pd.MultiIndex):
        unq_answers.index.names = top_values.index.names
    else:
        unq_answers.index.name = top_values.index.name
    return unq_answers

def numeric_values_as_keys(keys_w_all_answers, numeric_data):
    numeric_data_unq = numeric_data.dropna().unique()
    not_found = []
    mapped_answers = {}
    for nr in numeric_data_unq:
        if nr not in keys_w_all_answers:
            not_found.append(nr)
        else:
            mapped_answers[nr] = keys_w_all_answers[nr][0][1]
    if not_found:
        if len(not_found) <= 2 and mapped_answers:
            decision = 'Maybe'
        else:
            decision = 'False'
        decision += f'; found: {len(mapped_answers)}, not found: {len(not_found)}'
    elif mapped_answers:
        decision = True
    else:
        decision = False
    return decision, mapped_answers, not_found


def summarize_columns(df, available_since_v=7):
    summary_data = []
    #print('\n', len(df.columns), '\n', df.columns)
    # ic(df.columns)
    df['country code and phase + subversion'] = df['country code and phase'] + df['subversion_nr']
    
    for col_name in df.columns:
        # if col_name != 'source of drinking water' and col_name != 'type of toilet facility' and col_name != 'type of place of residence':
        #    continue
        col_data = df[col_name]
        col_data_v7 = df[df['version_nr'] >= available_since_v][col_name]
        col_summary = {"column_name": col_name}
        
        col_name_norm = replace_special_characters(col_name, col_mode=True)
        col_name_norm = remove_stopwords(col_name_norm)
        col_summary['column_name_normalized'] = col_name_norm 
        
        # Retrieve available and NA data
        available_data_count = col_data.dropna().count()
        len_data = len(col_data)
        available_data_count_v7 = col_data_v7.dropna().count()
        len_data_v7 = len(col_data_v7)
        
        col_summary["available_data amount"] = available_data_count
        col_summary["available_data [%]"] = round(available_data_count/len_data * 100, 3)
        #print(col_name, col_summary["available_data [%]"])
        #col_summary["na_data [%]"] = round(col_data.isna().sum()/len_data * 100, 3)
        if col_summary["available_data [%]"] <= 0.2:
            return
        
        for v in df['version_nr'].unique():
            col_summary[f'available_data amount V{v}'] = df[df['version_nr'] == v][col_name].dropna().count()
            col_summary[f'available_data [%] V{v}'] = round(col_summary[f'available_data amount V{v}']/len(df[df['version_nr'] == v]) * 100, 3)
            #col_summary[f'na_data [%] V{v}'] = round(df[df['version_nr'] == v][col_name].isna().sum()/len(df[df['version_nr'] == v]) * 100, 3)            
        
        col_summary["available_data V7+ amount"] = available_data_count_v7
        col_summary["available_data V7+ [%]"] = round(available_data_count_v7/len_data_v7 * 100, 3)
        #col_summary["na_data V7+ [%]"] = round(col_data_v7.isna().sum()/len_data_v7 * 100, 3)
        
        survey_counts = pd.DataFrame(columns=['indicator', 'percentage'])
        for survey in df['country code and phase + subversion'].unique():
            len_version = len(df[df['country code and phase + subversion'] == survey])
            perc = df[df['country code and phase + subversion'] == survey][col_name].count()/len_version *100
            survey_counts.at[survey, 'indicator'] = True if perc > 0 else False
            survey_counts.at[survey, 'percentage'] = perc
        col_summary['surveys with data in %'] = round(survey_counts['indicator'].sum()/len(survey_counts) * 100, 2)
        col_summary['surveys with data percentage data available'] = round(survey_counts[survey_counts['indicator'] == True]['percentage'].mean(), 2)
        
        numeric_data = pd.to_numeric(col_data, errors='coerce')
        string_data = col_data[numeric_data.isna()].dropna()
                
        if len(string_data) > 0:
            string_percentage = len(string_data) / len_data * 100
        else:
            string_percentage = 0
        numeric_percentage = len(numeric_data.dropna()) / len_data * 100
        
        col_summary["numeric [%]"] = round(numeric_percentage, 2)
        col_summary["string [%]"] = round(string_percentage, 2)    
        
        col_summary["unique numbers amount"] = numeric_data.dropna().nunique()
        
        replace_d = {}
        replace_d_w_keys = {}
        
        if len(numeric_data.value_counts()) < 150 and 0 in numeric_data.unique() and 1 in numeric_data.unique():
            if any([True for sd in string_data.unique() if sd[-3:] == 'yes']) or any([True for sd in string_data.unique() if sd[-3:] == 'yes']):
                if 2 in numeric_data.unique() or numeric_percentage > string_percentage:
                    for sd in string_data.unique():
                        if sd[-3:] == 'yes':
                            replace_d['yes'] = 1
                            replace_d_w_keys[sd] = 1
                        if sd[-2:] == 'no':
                            replace_d['no'] = 0
                            replace_d_w_keys[sd] = 0
                else:
                    for nr in numeric_data.unique():
                        if nr == 1:
                            replace_d[nr] = 'yes'
                            replace_d_w_keys[nr] = 'yes'
                        if nr == 0:
                            replace_d[nr] = 'no'
                            replace_d_w_keys[nr] = 'no'                 
                    
        if col_name in set(['case identification', 'version_nr', 'subversion_nr', 
                            'country code and phase', 'country code and phase + subversion', 'generated clnr from case id']):
            #ic('passing', col_name)
            pass
        
        elif len(string_data) > 0:
            # If the column is of string data type
            #ic('hi', col_name)
            unq_answers_wo_keys = False
            col_summary["data_type"] = "categorical"
            value_counts = string_data.value_counts(normalize=True) * 100
            value_counts = value_counts.round(2)
            top_values = value_counts #.head(200)  # Get top 100 most common answers
            col_summary["unique_answers"] = top_values
            #ic(top_values)
            dubious_answers, corrected_answers, corrected_answers_w_keys, keys_w_all_answers = autocorrect(top_values.index.tolist(), col_summary)
            col_summary["dubious_answers"] = dubious_answers if dubious_answers else np.NaN
            corrected_answers = {**replace_d, **corrected_answers}
            corrected_answers_w_keys = {**replace_d_w_keys, **corrected_answers_w_keys}
            #ic(corrected_answers, type(corrected_answers))
            #ic(corrected_answers_w_keys, type(corrected_answers_w_keys))
            if corrected_answers_w_keys:
                col_data = col_data.replace(corrected_answers_w_keys)
                numeric_data = pd.to_numeric(col_data, errors='coerce')
                string_data = col_data[numeric_data.isna()].dropna()

            if len(string_data) > 0:
                value_counts = string_data.value_counts(normalize=True) * 100
                value_counts = value_counts.round(2)
                top_values = value_counts #.head(200)
                unq_answers_wo_keys = create_unq_answers(top_values)

                if not isinstance(unq_answers_wo_keys, pd.Series):
                    raise TypeError('unq_answers_wo_keys is not a pd.Series', unq_answers_wo_keys, type(unq_answers_wo_keys))
                if col_name == 'source of drinking water':
                    ic(unq_answers_wo_keys, type(unq_answers_wo_keys))

                col_summary["unique_answers wo keys"] = unq_answers_wo_keys if len(unq_answers_wo_keys) > 0 else np.NaN

                col_summary["unique_answers amount"] = len(unq_answers_wo_keys)            
            col_summary["unique_answers amount"] = len(value_counts)
            
            if corrected_answers and not pd.isna(corrected_answers):
                #ic(corrected_answers, type(corrected_answers))
                col_summary['replace_d corrected_answers'] = corrected_answers
                col_summary['replace_d corrected_answers w keys'] = corrected_answers_w_keys
                  
                if len(string_data) > 0:
                    string_percentage = len(string_data) / len_data * 100
                else:
                    string_percentage = 0
                numeric_percentage = len(numeric_data.dropna()) / len_data * 100
                
                col_summary["numeric [%]"] = round(numeric_percentage, 2)
                col_summary["string [%]"] = round(string_percentage, 2)    
            else:
                col_summary["data_type"] = "numeric"
            
            if unq_answers_wo_keys is not False and len(unq_answers_wo_keys) > 0: 
                #ic('in unique_answers_wo_keys')   
                unq_answer_d = {}
                #ic(unq_answers_wo_keys)
                for unq_a in unq_answers_wo_keys.index.tolist():
                    unq_a2 = replace_special_characters(unq_a, remove_brackets=False)
                    unq_a2 = remove_stopwords(unq_a2)
                    unq_answer_d[unq_a] = unq_a2
                if unq_answer_d:
                    #ic('hi')
                    col_summary["unique_answers wo keys list"] = list(unq_answer_d.values())
                    col_summary["unique_answers wo keys dict"] = unq_answer_d
                else:
                    col_summary["unique_answers wo keys list"] = np.NaN
                    col_summary["unique_answers wo keys dict"] = np.NaN     
            
            if len(numeric_data) > 0:
                ###check for numeric values which might be keys   
                     
                decision, mapped_answers, none_found_keys = numeric_values_as_keys(keys_w_all_answers, numeric_data)   
                col_summary['replace_d numeric values as keys'] = mapped_answers if mapped_answers else np.NaN
                col_summary['numeric values as keys decision'] = decision if decision else np.NaN
                col_summary['numeric values as keys not found'] = none_found_keys if none_found_keys else np.NaN
                #ic(col_name, decision, mapped_answers, none_found_keys)
                
        # Test for data type
        if col_summary["numeric [%]"]:
            # If the column is of numeric data type
            col_summary["data_type"] = "numeric"
            col_summary["max"] = numeric_data.max()
            col_summary["min"] = numeric_data.min()
            col_summary["mean"] = round(numeric_data.mean(), 3)
            col_summary["std"] = round(numeric_data.std(), 3)     
        if col_summary["numeric [%]"] and col_summary["string [%]"]:
            # If the column has mixed data type
            col_summary["data_type"] = "mixed"

        #ic(col_name_norm)
        summary_data.append(col_summary)
    
    summary_df = pd.DataFrame(summary_data)
    #ic(summary_df[['column_name', "available_data [%]"]])
    return summary_df



def loading_wrapper(start_nr, cols_to_load_simultaneously, file_paths, dataset_type, folder_path, col_labels, create_GEID=False, create_DHSID=False, 
                    max_workers=47, load_all=False, create_version_nr=False):
    if not load_all:
        if cols_to_load_simultaneously == 'all':
            col_labels = col_labels[start_nr:]
        else:
            col_labels = col_labels[start_nr:start_nr+cols_to_load_simultaneously]
    # ic(col_labels)
           
    all_data, meta_data = load_data(file_paths, dataset_type, folder_path, col_labels=col_labels, create_GEID=create_GEID, create_DHSID=create_DHSID, 
                                    max_workers=max_workers, create_version_nr=create_version_nr)
    df = combine_survey_dfs(all_data)[0]
    return df


def summary_df_loading_wrapper(start_nr, cols_to_load_simultaneously, file_paths, dataset_type, folder_path, all_col_labels, create_GEID=False, create_DHSID=False,
                               available_since_v=7, max_workers=1):

    df = loading_wrapper(start_nr, cols_to_load_simultaneously, file_paths, dataset_type, folder_path, all_col_labels, create_GEID=create_GEID, 
                         create_DHSID=create_DHSID, max_workers=max_workers, create_version_nr=True)
    df = df.dropna(axis=1, how='all')
    summary_df = summarize_columns(df, available_since_v=available_since_v)
    return summary_df

def create_summary_df(file_paths, dataset_type, all_col_labels, cols_to_load_simultaneously, folder_path, create_GEID=False, create_DHSID=False, available_since_v=7):
    #all_col_labels = ['has television']
    # all_col_labels = [l for l in all_col_labels if 'salt' in l or 'sel' in l]
    #cols_to_load_simultaneously = 1
    # ic('hi')
    # only_load = ['source of drinking water', 'type of toilet facility', 'type of place of residence']
    # if only_load:
    #     all_col_labels = only_load
    
    start_nr_l = list(range(0, len(all_col_labels), cols_to_load_simultaneously))
    # ic(all_col_labels, start_nr_l)
    partial_summary_df_loading_wrapper = partial(summary_df_loading_wrapper, file_paths=file_paths, dataset_type=dataset_type, folder_path=folder_path, all_col_labels=all_col_labels, 
                                                 available_since_v=available_since_v, cols_to_load_simultaneously=cols_to_load_simultaneously, 
                                                 create_GEID=create_GEID, create_DHSID=create_DHSID, max_workers=1)
    with ProcessPoolExecutor(max_workers=47) as executor:
        results = list(tqdm(executor.map(partial_summary_df_loading_wrapper, start_nr_l), total=len(start_nr_l)))
    summary_df = pd.concat(results, axis=0)
    
    ###Needs to be corrected since not all dfs have the same length
#     summary_df['available_data [%]'] = round((summary_df['available_data amount']/summary_df['available_data amount'].max()) * 100
#     summary_df["na_data [%]"] = 100 - summary_df['available_data [%]']
#     summary_df['available_data V7+ [%]'] = (summary_df['available_data V7+ amount']/summary_df['available_data V7+ amount'].max()) * 100
#     summary_df["na_data V7+ [%]"] = 100 - summary_df['available_data V7+ [%]']
    #Some more manipulations
    filtered_df = summary_df[(summary_df['column_name'] == 'country code and phase') | (summary_df['column_name'] == 'version_nr')
                             | (summary_df['column_name'] == 'subversion_nr') | (summary_df['column_name'] == 'country code and phase + subversion')
                             | (summary_df['column_name'] == 'generated hhnr from case id') | (summary_df['column_name'] == 'generated clnr from case id')]
    s_df = filtered_df.drop_duplicates(subset='column_name', keep='first')
    summary_df = summary_df[(summary_df['column_name'] != 'country code and phase') & (summary_df['column_name'] != 'version_nr') & 
                            (summary_df['column_name'] != 'subversion_nr') & (summary_df['column_name'] != 'country code and phase + subversion')
                            & (summary_df['column_name'] != 'generated hhnr from case id') & (summary_df['column_name'] != 'generated clnr from case id')]
    summary_df = pd.concat([s_df, summary_df], axis=0)
    summary_df = summary_df.sort_index()
    return summary_df


#@cache_to_sqlite('my_cache.db')
def replace_special_characters(s, remove_brackets=True, col_mode=False):
    """
    Replace special characters in a string with a space,
    and replace multiple spaces with a single space.

    Args:
        s (str): The input string.
        
    Returns:
        str: The cleaned string.
    """
    #remove stuff in brackets
    if remove_brackets:
        if '(' in s and ')' in s:
            s = s.split('(')[0] + s.rsplit(')', 1)[1]
        if '[' in s and ']' in s:
            s = s.split('[')[0] + s.rsplit(']', 1)[1]
            
    if col_mode:
        for spl in [': ', ' - ']:
            if spl in s:
                s = s.split(spl, 1)
                # ic('splitted col name', s)
                s = s[1]
                break
    # Define a regular expression pattern for the special characters
    # if not drop_special:
    #     pattern = global_pattern_special  # Modified pattern
    # else:
    pattern = global_pattern  # Corrected pattern (removed trailing comma)
    
    # Use re.sub() to replace the special characters with a space
    result = re.sub(pattern, ' ', s)
    
    # Replace multiple spaces with a single space
    result = re.sub(r'\s+', ' ', result)
    if result[-1] == ' ':
        result = result[:-1]
        
    for sp in [' don t', ' won t', ' can t']:
        sp_replace = sp[:-2] + "'" + sp[-1:]
        if sp in result or sp[1:] == result[:len(sp[1:])]:
            if sp[1:] + ' ' in result or sp[1:] == result[-len(sp[1:]):]:
                if sp in result:
                    result = result.replace(sp, sp_replace)
                else:
                    result = result.replace(sp[1:], sp_replace[1:])
    
    return result

 
#@cache_to_sqlite('my_cache.db')
def remove_stopwords(s):
    if not ' ' in s:
        return s
    
    word_l = s.split(' ')
    #remove stopwords
    stopwords = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'is', 'it', 'of', 'as', 's',
    'own', 'owns', 'has', 'by', 'for', 'goods', 'do', 'does', 'did', 
    'have', 'has', 'had', 'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 
    'that', 'those', 'these', 'this', 'those', 'these', 'those', 'this', 'those', 'these', 'this',
    'at', 'on', 'from', 'between', 'be', 'been', 'being', 
    'household', 'hh', 'hhousehold', 'househ', 
    'na', 'de', 'usable', 'how', 'many', 'members', 'member', 
    'cs', 'na'}

    word_l = [word for word in word_l if word not in stopwords]
    word_l = list(dict.fromkeys(word_l))
    
    if not word_l:
        return s
    elif len(word_l) == 1:
        s = word_l[0]
    else:
        s = ' '.join([word for word in word_l])    
    return s


#@profile
def overwriting_percentage_and_decision(df_in, base_col, row_col, row, integrated_col):
    # Initialize default values
    row['original different value counts'] = np.NaN
    row['original % overwriting (as % of col2)'] = np.NaN
    row['original % different values (as % of col2)'] = np.NaN
    
    row['integrated different value counts'] = np.NaN
    row['integrated % overwriting (as % of col2)'] = np.NaN
    row['integrated % different values (as % of col2)'] = np.NaN

    l = [df_in]
    if integrated_col is not False:
        l.append(integrated_col)
    #ic(df_in.head(5))
    # if not 'salt' in base_col:
    #     return row, False, False

    for nr, df in enumerate(l):
        pre = 'original ' if nr == 0 else 'integrated '
        
        df_different = df[df[base_col] != df[row_col]][[base_col, row_col]]
        #drop rows with na
        df_different = df_different.dropna()
        
        try:        
            row[pre + '% overwriting (as % of col2)'] = round(len(df[(df[base_col].notna()) & (df[row_col].notna())])
                                                                / len(df[df[row_col].notna()]) * 100, 2)
            row[pre + '% different values (as % of col2)'] = round(len(df_different) / len(df[df[row_col].notna()]) * 100, 2)
        except ZeroDivisionError as e:
            ic('df', df)
            ic('len not na', len(df[df[row_col].notna()]))
            raise e
        
        if row[pre + '% different values (as % of col2)']:
            try:
                vc = df_different.value_counts().iloc[:10,]
                row[pre + 'different value counts'] = vc
            except ValueError as e:
                ic("Error processing df_different:", df_different)
                warn(e)

    #Make an automatic decision - Needs to be validated manually!!!
    #to do
    if pd.isna(row['original % different values (as % of col2)']) or \
            row['original % different values (as % of col2)'] == 0:       
        similarity_indicator = 'True'
    elif pd.notna(row['original % different values (as % of col2)']) and \
            row['original % different values (as % of col2)'] <= 10:
        similarity_indicator = 'Maybe'
    elif pd.notna(row['original % different values (as % of col2)']) and \
            row['original % different values (as % of col2)'] <= 25:
        similarity_indicator = 'Probably not'
    else:
        similarity_indicator = 'False'
    
    if row['cosine string similarity columns'] < 75:
        similarity_indicator += '? Probably not!?'
    elif row['cosine string similarity columns'] < 85:
        similarity_indicator += '? Maybe!?'
    elif row['cosine string similarity columns'] < 95:
        similarity_indicator += '? SC'
    
    if pd.notna(row['cosine string similarity answers']) and \
            row['cosine string similarity answers'] < 92:
        if 'True' == similarity_indicator[:4]:
            similarity_indicator += \
                '? Cosine str similarity of answers (indicator, %): ' + row['cosine string similarity indicator'] + ', ' + str(row['cosine string similarity answers']) + '% '
        else:
            similarity_indicator += ' Cosine str similarity of answers (indicator, %): ' + row['cosine string similarity indicator'] + ', ' + str(row['cosine string similarity answers']) + '% '
    
    elif pd.notna(row['cosine string similarity answers']) and \
            row['cosine string similarity answers'] < 99:
        similarity_indicator += \
            ' Cosine str similarity of answers (indicator, %): ' + row['cosine string similarity indicator'] + ', ' + str(row['cosine string similarity answers']) + '% '
        
    if row['original % overwriting (as % of col2)'] > 0:
        similarity_indicator += (f' Caution! Overwriting: {row["original % overwriting (as % of col2)"]} diff values: '
            f'{row["original % different values (as % of col2)"]}')

    if row['cosine string similarity columns'] < 98:
        similarity_indicator += ' Cosine str similarity of columns: ' + str(row['cosine string similarity columns']) + '%'    
    
    if row['datatype similarity'] is not True and pd.notna(row['datatype similarity']):
        similarity_indicator += f' -- Caution! datatype similarity: {row["datatype similarity"]}'
        if 'True' == similarity_indicator[:4]:
            similarity_indicator = 'True? DT ' + similarity_indicator[4:]
            
    if row['numeric similarity'] is not True and pd.notna(row['numeric similarity']):
        similarity_indicator += f' -- Caution! numeric data similarity: {row["numeric similarity"]}'
        if 'True' == similarity_indicator[:4]:
            similarity_indicator = 'True? NS ' + similarity_indicator[4:]
        
    if row['integrated % overwriting (as % of col2)'] > 0 :# and not np.isnan(row['integrated % different values (as % of col2)']):
        similarity_indicator += (f' -- Caution! overwriting in integrated! '
            f'{row["integrated % overwriting (as % of col2)"]} different values: {row["integrated % different values (as % of col2)"]} % of col2 data')
    elif row['integrated % different values (as % of col2)'] > 0 :# and not np.isnan(row['integrated % different values (as % of col2)']):
        similarity_indicator += (f' -- Caution! overwriting in integrated! '
            f'{row["integrated % overwriting (as % of col2)"]} different values: {row["integrated % different values (as % of col2)"]} % of col2 data')
    
    if pd.isna(row['original % different values (as % of col2)']) or \
            row['original % different values (as % of col2)'] == 0:
        if integrated_col is False:
            integrated_col = df_in.copy()
        integrated_col[base_col] = integrated_col[base_col].fillna(integrated_col[row_col])
    
    #if automatic decision was unsure, use this row to potentially run the comparison again (from these rest/unsure rows)
    new_row = row.copy()
    new_row['similar_columns nr'] = new_row['similar_columns nr'] + 0.01
    for key in ['original different value counts',
                'original % overwriting (as % of col2)',
                'original % different values (as % of col2)',
                'integrated different value counts',
                'integrated % overwriting (as % of col2)',
                'integrated % different values (as % of col2)',
                'datatype similarity','numeric similarity', 
                'cosine string similarity columns', 'cosine string similarity answers', 'cosine string similarity indicator']:
        new_row[key] = np.NaN
 
    row['overall indicator'] = similarity_indicator
    row['overall indicator short'] = similarity_indicator[:5]
    #ic(base_col, row_col, similarity_indicator, row['original % overwriting (as % of col2)'], row['original % different values (as % of col2)'], df_different.head(5))
    return row, integrated_col, new_row

#@profile
def process_group(group_df, file_paths, dataset_type, folder_path, input_dir):
    # Assume the first row is the one we're comparing others against
    base_row = group_df.iloc[0]
    
    #skip if base_row is version_nr or 100% available or < 0.7% available
    if base_row["column_name"] == 'version_nr' or base_row["available_data [%]"] >= 99.9:
        # return None and rest rows
        return None, group_df.iloc[1:]
    # if base_row['column_name'] != 'result of salt test for iodine':
    #     return

    client = OpenAI()
    
    results = []
    rest_group_rows = []
    integrated_col = False

    # Iterate through the rest of the rows in this group
    # ic('group', group_df)
    col_labels = group_df['column_name'].to_list()
    all_data, _ = load_data(file_paths, dataset_type, folder_path, col_labels=col_labels, 
                                    create_GEID=False, create_DHSID=False, max_workers=1, 
                                    create_version_nr=False, load_keys=False)
    actual_values_df = combine_survey_dfs(all_data)[0]
    
    #ic(actual_values_df.head(5))
    for index, row in group_df.iterrows():
        if index == base_row.name:  # Skip the base row
            results.append(base_row)  # Append base row as it is
            continue
        
        new_row = row.copy()
        new_row['cosine string similarity indicator'] = np.NaN
        new_row['cosine string similarity answers'] = np.NaN

        base_col = base_row['column_name']
        row_col = new_row['column_name']
        base_col_norm, row_col_norm = base_row['column_name_normalized'], new_row['column_name_normalized']
        similarity = cosine_similarity(base_row['ada_embedding'].reshape(1, -1), new_row['ada_embedding'].reshape(1, -1))
        similarity = similarity[0][0] * 100
        new_row['cosine string similarity columns'] = similarity
        if similarity < 79:
            new_row['overall indicator'] = f'False: cosine similarity of column names to low {similarity}' 
            results.append(new_row)
            rest_group_rows.append(new_row)
            ic('not processing group', base_col, row_col, similarity, base_col_norm, row_col_norm)
            continue
        #ic('processing group', base_col, row_col, similarity, base_col_norm, row_col_norm)        
        # ic('process_group', base_col, row_col)
        
        try:
            base_values, row_values = False, False
            if isinstance(base_row['unique_answers wo keys dict'], dict):
                base_values = base_row['unique_answers wo keys dict']
            if isinstance(new_row['unique_answers wo keys dict'], dict):
                row_values = new_row['unique_answers wo keys dict']
            if base_values and '' in base_values:
                del base_values['']
            if row_values and '' in row_values:
                del row_values['']
            if base_values == {}:
                base_values = False
            if row_values == {}:
                row_values = False
        except Exception as e:
            ic(base_col, row_col)
            ic(base_row['unique_answers wo keys dict'])
            ic(new_row['unique_answers wo keys dict'])
            ic(base_values, row_values)
            
        if isinstance(base_values, float) and np.isnan(base_values):
            ic('base_values NaN', base_values)
            base_values = False
        if isinstance(row_values, float) and np.isnan(row_values):
            ic('row_values NaN', row_values)
            row_values = False
        
        new_row['datatype similarity'] = False
        new_row['numeric similarity'] = np.NaN
        
        #determine datatype similarity and numeric similarity
        if base_values and row_values:
            if not np.isnan(base_row["mean"]) and not np.isnan(new_row["mean"]):
                new_row['datatype similarity'] = True
            elif np.isnan(base_row["mean"]) and np.isnan(new_row["mean"]):
                new_row['datatype similarity'] = True
        elif not base_values and not row_values:
            if not np.isnan(base_row["mean"]) and not np.isnan(new_row["mean"]):
                new_row['datatype similarity'] = True
            elif np.isnan(base_row["mean"]) and np.isnan(new_row["mean"]):
                new_row['datatype similarity'] = True
                
        if base_row['numeric [%]'] > 0 and new_row['numeric [%]'] > 0:
            if new_row["mean"] + new_row["std"] > base_row["mean"] > new_row["mean"] - new_row["std"]:
                new_row['numeric similarity'] = True
            elif new_row["mean"] + 2 * new_row["std"] > base_row["mean"] > new_row["mean"] - 2 * new_row["std"]:
                new_row['numeric similarity'] = 'Maybe '
            else:
                new_row['numeric similarity'] = 'False '    
                
            new_row['datatype similarity'] = True
            if (isinstance(new_row['unique_answers wo keys'], pd.Series) and 
                    len(new_row['unique_answers wo keys']) > 0) or \
                    ((isinstance(base_row['unique_answers wo keys'], pd.Series) and 
                    len(base_row['unique_answers wo keys']) > 0)):
                new_row['datatype similarity'] = 'Maybe'
            
                
        if base_values is not False and row_values is not False:    
            cos_str_sim, answer_mapping, non_matched_answers = calculate_cosine_similarities_for_integration_candidate(base_values, 
                                                                          row_values, base_col_norm, 
                                                                          row_col_norm, input_dir, client)
            new_row['answer mapping for integration'] = answer_mapping
            new_row['answer mapping for integration not found'] = non_matched_answers
            new_row['cosine string similarity answers'] = cos_str_sim
            new_row['cosine string similarity indicator'] = 'False'
            if new_row['cosine string similarity answers'] >= 70:
                new_row['cosine string similarity indicator'] = 'Probably not'
            if new_row['cosine string similarity answers'] >= 80:
                new_row['cosine string similarity indicator'] = 'Maybe'
            if new_row['cosine string similarity answers'] >= 90:
                new_row['cosine string similarity indicator'] = 'Probably'
            if new_row['cosine string similarity answers'] >= 95:
                new_row['cosine string similarity indicator'] = 'Most Probably'
            if new_row['cosine string similarity answers'] >= 98:
                new_row['cosine string similarity indicator'] = 'True'            
            #ic(cos_str_sim, new_row['cosine string similarity indicator'], answer_mapping)
                        
        new_row, integrated_col, rest_group_row = overwriting_percentage_and_decision(actual_values_df, base_col, row_col, new_row, integrated_col)
        # Append the new row to the results list
        results.append(new_row)
        if rest_group_row is not False:
            for r in rest_group_rows:
                if r['column_name'] == rest_group_row['column_name']:
                    warn(f'double row in rest_group_rows {r["column_name"]}')
                    break
            else:
                rest_group_rows.append(rest_group_row)
        

    # Create a new DataFrame from the results list
    updated_group_df = pd.DataFrame(results)
    rest_group_df = None
    if rest_group_rows:
        rest_group_df = pd.DataFrame(rest_group_rows)
    return updated_group_df, rest_group_df


def cluster_hdbscan(embeddings, method='eom', alpha=1.0, cluster_selection_epsilon=0.2, min_cluster_size=2, min_samples=2):
    # Normalize the embeddings
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(list(embeddings))
    # Apply HDBSCAN with parameters suited for smaller clusters
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,  # Allows for 2-point clusters
        cluster_selection_epsilon=cluster_selection_epsilon,  # Lower value to prevent merging of distinct clusters
        cluster_selection_method=method, # 'leaf' to get smaller clusters
        alpha=alpha,                        # Increase alpha for more conservative clustering ## default 1.0
        min_samples=min_samples)
    cluster_labels = clusterer.fit_predict(normalized_embeddings)
    return cluster_labels


def cluster_kmeans(embeddings, n_clusters=3):
    """
    Cluster embeddings using the KMeans algorithm with a specified number of clusters.

    Parameters:
    embeddings (iterable): The embeddings to be clustered.
    n_clusters (int): The number of clusters to form.

    Returns:
    cluster_labels (array): Index of the cluster each sample belongs to.
    """
    
    # Normalize the embeddings
    scaler = StandardScaler()
    #ic(embeddings)
    normalized_embeddings = scaler.fit_transform(embeddings)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(normalized_embeddings)
    
    return cluster_labels

def sanitize_filename(text):
    # Replace or remove special characters
    # Keeps alphanumeric, underscores, hyphens, and periods
    return re.sub(global_pattern_folders, '_', text)

def save_embedding_to_file(embedding, text, cache_dir):
    folder_path = os.path.join(cache_dir, text[:3])
    os.makedirs(folder_path, exist_ok=True)
    filename = os.path.join(folder_path, f"{text}.npy")
    if not isinstance(embedding, np.ndarray):
        ic(text)
        warn('embedding is not a np.ndarray', embedding, type(embedding))
    np.save(filename, embedding)

def load_embedding_from_file(text, cache_dir):
    filename = os.path.join(cache_dir, text[:3], f"{text}.npy")
    if os.path.exists(filename):
        try:
            return np.load(filename, allow_pickle=True)
        except Exception as e:
            ic(text, filename, e)
    return None


def get_embedding(text_in, client, col, input_dir, model="text-embedding-ada-002", get_single_word_embeddings=False):
    orig_a, norm_a = text_in
    if not norm_a:
        ic(col)
        ic(norm_a)
        ic(orig_a)
        warn(f'norm_a is empty {norm_a}, setting to {orig_a}')
        norm_a = orig_a
        if not norm_a:
            ic(col, norm_a, text_in)
            warn(f'normalized answer is empty {norm_a} {text_in}')
            return
    text = norm_a.replace("\n", " ")
    text_fn = sanitize_filename(text)
    
    embedding = load_embedding_from_file(text_fn, input_dir + '/embeddings_cache/')
    if embedding is None:
        embedding = np.array(client.embeddings.create(input=[text], model=model).data[0].embedding)
        save_embedding_to_file(embedding, text_fn, input_dir + '/embeddings_cache/')
    if get_single_word_embeddings:
        words = text.split(' ')
        if '' in words:
            words.remove('')
        if not words:
            return embedding, [embedding], [text]
        word_embeddings = []
        for word in words:
            word_fn = sanitize_filename(word)
            word_embedding = load_embedding_from_file(word_fn, input_dir + '/embeddings_cache/')
            if word_embedding is None:
                try:
                    word_embedding = np.array(client.embeddings.create(input=[word], model=model).data[0].embedding)
                except Exception as e:
                    ic(text, words, word, e)
                save_embedding_to_file(word_embedding, word_fn, input_dir + '/embeddings_cache/')
            word_embeddings.append(word_embedding)
        return embedding, word_embeddings, words
    return embedding


def retrieve_embeddings(df, input_dir):
    client = OpenAI()
    embeddings = []

    for col in df['column_name_normalized']:
        embedding = get_embedding(('_', col), client, col, input_dir, model='text-embedding-ada-002')
        embeddings.append(embedding)
    
    df['ada_embedding'] = embeddings
    
    cluster_labels = cluster_hdbscan(embeddings, method='eom')
    # Add the cluster labels to your DataFrame
    df['similar_columns nr'] = cluster_labels

    return df

#@cache_to_sqlite('my_cache.db')
#@profile
def calculate_cosine_similarities_old(answers_base, answers_row, col_base, col_row, input_dir):
    client = OpenAI()
    embeddings_row = []
    for (answer_row, ar_norm) in answers_row.items():
        embedding = get_embedding((answer_row, ar_norm), client, col_row, input_dir, model='text-embedding-ada-002')
        embeddings_row.append(embedding)
    embeddings_base = []
    for (answer_base, ab_norm) in answers_base.items():
        embedding = get_embedding((answer_base, ab_norm), client, col_base, input_dir, model='text-embedding-ada-002')
        embeddings_base.append(embedding)
    similarities_ges = []
    answer_mapping = {}
    non_matched_answers = []
    for embedding_row, (answer_row, ar_norm) in zip(embeddings_row, answers_row.items()):
        similarities = {}
        for embedding_base, (answer_base, ab_norm) in zip(embeddings_base, answers_base.items()):
            similarity = cosine_similarity(embedding_row.reshape(1, -1), embedding_base.reshape(1, -1))
            similarities[(answer_row, answer_base)] = similarity * 100
            #negations might be to close to the same non negated answer
            if negation_test(answer_row) != negation_test(answer_base):
                similarities[(answer_row, answer_base)] = similarity * 100 - 30
            #ic(answer_row, answer_base, ar_norm, ab_norm, similarity * 100)
        similarities_ges.append(max(similarities.values()))
        if 80 < max(similarities.values()) < 99.99999:
            answer_mapping[answer_row] = max(similarities, key=similarities.get)[1]
        elif max(similarities.values()) < 80:
            non_matched_answers.append(answer_row)
    #ic(answer_mapping)
    return np.mean(similarities_ges), answer_mapping, non_matched_answers


def calculate_cosine_similarities_for_integration_candidate(answers_base, answers_row, col_base, col_row, input_dir, client):
    
    # Get all Similarities and embeddings
    answers_to_words = {}
    word_to_index = {}
    all_embeddings = {}
    for nr, col_name in enumerate([col_base, col_row]):
        try:
            # Calculate question embedding
            col_name_embedding, embeddings, col_name_split = get_embedding((col_name, col_name), client, col_name, input_dir, model='text-embedding-ada-002', get_single_word_embeddings=True)
            word_to_index[col_name] = len(all_embeddings)  
            all_embeddings[col_name] = col_name_embedding
            answers_to_words[col_name] = col_name_split
        except Exception as e:
            ic(col_name)
            raise(e)



    # Process base answers
    for answer_base, ab_norm in answers_base.items():
        answer_embedding, embeddings, ab_norm_split = get_embedding((answer_base, ab_norm), client, col_base, input_dir, model='text-embedding-ada-002', get_single_word_embeddings=True)
        if answer_base not in word_to_index:
            word_to_index[answer_base] = len(all_embeddings)
            all_embeddings[answer_base] = answer_embedding
        answers_to_words[answer_base] = ab_norm_split
        if len(ab_norm_split) != len(embeddings):
            ic(ab_norm_split, len(embeddings))
        assert len(ab_norm_split) == len(embeddings)
        for word, embedding in zip(ab_norm_split, embeddings):
            if word not in word_to_index:
                word_to_index[word] = len(all_embeddings)
                all_embeddings[word] = embedding

    # Process row answers
    for answer_row, ar_norm in answers_row.items():
        answer_embedding, embeddings, ar_norm_split = get_embedding((answer_row, ar_norm), client, col_row, input_dir, model='text-embedding-ada-002', get_single_word_embeddings=True)
        if answer_row not in word_to_index:
            word_to_index[answer_row] = len(all_embeddings)
            all_embeddings[answer_row] = answer_embedding
        answers_to_words[answer_row] = ar_norm_split
        if len(ar_norm_split) != len(embeddings):
            ic(ar_norm_split, len(embeddings))
        assert len(ar_norm_split) == len(embeddings)
        for word, embedding in zip(ar_norm_split, embeddings):
            if word not in word_to_index:
                word_to_index[word] = len(all_embeddings)
                all_embeddings[word] = embedding

    # Create similarity matrix
    words_l = list(word_to_index.keys())
    # ic(len(words_l), len(all_embeddings), len(word_to_index))
    similarity_df = pd.DataFrame(cosine_similarity(list(all_embeddings.values())), index=words_l, columns=words_l) * 100

    # Compute similarities
    answer_mapping = {}
    non_matched_answers = []
    best_similarities = []
    for answer_row in answers_row:
        best_similarity = 0
        best_match = None

        for answer_base in answers_base:
            similarity_info = retrieve_similarity(answer_row, answer_base, similarity_df, answers_to_words, col_row, col_base)
            similarity = similarity_info[0]  # avg_max_similarity

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = answer_base

        best_similarities.append(best_similarity)
        
        # Categorize based on similarity threshold
        if best_similarity > 85:
            answer_mapping[answer_row] = best_match
        else:
            non_matched_answers.append(answer_row)

    return np.mean(best_similarities), answer_mapping, non_matched_answers


#@profile
def process_group_wrapper(integration_summary_df, file_paths, dataset_type, folder_path, input_dir):
    # Group the data by similar_columns nr
    grouped = integration_summary_df.groupby('similar_columns nr')
    #embeddings_cache = load_embedding_from_file(input_dir + 'embeddings_cache.pkl')
    
    group_l = [group for name, group in grouped]
    for group in group_l:
        if group['column_name'].nunique() != len(group):
            ic('non unq', group)
    
    ic('\n\n\n\n')
    cut_perc = 10
    results_ges = []
    nr = 0
    while group_l:
        new_group_l = []
        for group in group_l:
            if len(group) >= 2 and ((group['available_data [%]'].sum()) >= cut_perc or \
                    ('available_data V7+ [%]' in group and group['available_data V7+ [%]'].sum() >= cut_perc)):
                if len(group[(group['column_name'] == 1) | (group['column_name'] == '1')]) >= 1:
                    ic('still happens group')
                    # drop this row
                    group = group[(group['column_name'] != 1) & (group['column_name'] != '1')]
                
                #sort                
                group = group.sort_values(by=['available_data [%]'], ascending=False)
                new_group_l.append(group)
    
        if False:
            results = []
            rest_groups_l = []
            for group in new_group_l:
                res, rest_group = process_group(group, file_paths, dataset_type, folder_path, input_dir)
                results.append(res)
                rest_groups_l.append(rest_group)
            group_l = rest_groups_l
        else:
            process_group_partial = partial(process_group, file_paths=file_paths, dataset_type=dataset_type, folder_path=folder_path, input_dir=input_dir)
            # Create a ProcessPoolExecutor to process the groups in parallel
            try:
                with ProcessPoolExecutor(max_workers=47) as executor:
                    # Use executor.map to apply the process_group function to each group                        
                    results = list(tqdm(executor.map(process_group_partial, new_group_l), total=len(new_group_l)))
                    # ic(results)
                    results = [res for res in results if isinstance(res, tuple)]
                    group_l = [item[1] for item in results if item[1] is not None]
                    results = [item[0] for item in results if item[0] is not None]
                    results_ges.extend(results)
            except Exception as e:
                raise(e)    
            nr += 1
    result_df = pd.concat(results_ges, axis=0, join='outer')
    return result_df


def unify_answers_iterator_wrapper(df, input_dir):

    df = df.reset_index(drop=True)

    # Execute in parallel and collect results
    partial_unify_answers_iterator = partial(unify_answers_iterator, input_dir=input_dir)

    with ThreadPoolExecutor(max_workers=47) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(partial_unify_answers_iterator, row) for index, row in df.iterrows()]
        results = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            #try:
            result = future.result()
            results.append(result)
            #except Exception as e:
             #   print(f"Error: {e}")

    # Reconstruct DataFrame from results
    results = [r for r in results if r is not None]
    df = pd.DataFrame([r for r in results], index=[r.name for r in results])
    df = df.sort_index()
    #df.update(df_copy)
    return df


class CustomError(Exception):
    """Base class for other exceptions"""
    pass


def unify_answers_iterator(row, input_dir):
    skip_rows = set(['country code and phase', 'sample strata for sampling errors', 'region', 
            'stratification used in sample design', 
            'version_nr', 'subversion_nr', 'case identification', 'district', 'country code and phase + subversion',
            'interviewer identification', 'nearest health facility', 'sample stratum number', 
            'code for nearest health facility', 'province', 'district of residence', 'keyer identification', 'county', 'municipality', 'district code',
            'dpartement', 'county - as was collected', 'county - as in the final tables'
            'bar code for hiv blood sample', 'department', 'state', 'province of residence',  "interviewer's assigned state", 
            '2008 state codes reclassified into 2003 state codes', 'state name', 'state of residence', 'office editor', 'pool', 'household district'
            'governorate', 'original district number', 'new province'
            'governorates', 'prefecture/canton', 'team number', 'region of residence', 'domain', 'regions exclud zanzibar/pemba', 'region (excluding zanzibar)',
            'generated clnr from case id', 'generated hnr from case id'
            ])
 
    row = row.copy()
    cut_at_perc = 0.1
    #embeddings_to_answers = {}  # Maps embeddings to answers
    answers_to_embeddings = {}  # Maps answers to embeddings
            

    if (row['available_data [%]'] < cut_at_perc) or (row['string [%]'] == 0) \
            or (np.isnan(row['string [%]'])) \
            or (row['column_name'] in skip_rows) or not isinstance(row['unique_answers wo keys'], pd.Series):
        return row

    # if row['column_name'] not in ['source of drinking water', 'type of toilet facility', 'type of place of residence']:
    #    return
    
    # ic(row['column_name'])
    client = OpenAI()

    # Retrieve embeddings and cluster them
    answers_to_embeddings = {}
    for answer_orig, answer_norm in row['unique_answers wo keys dict'].items():
        embedding, single_word_embeddings, words_l = get_embedding((answer_orig, answer_norm), client, row['column_name'], 
                                                          input_dir, model='text-embedding-ada-002', get_single_word_embeddings=True)
        answers_to_embeddings[(answer_orig, answer_norm)] = (embedding, single_word_embeddings)

    # HDBSCAN clustering
    client = OpenAI()
    if answers_to_embeddings and len(answers_to_embeddings) > 3:
        value_counts = row['unique_answers wo keys']
        subs_nr = 0
        insert_nr = 0
        penalty_for_uneven_word_count_l = [20, 5, 2, 0, 0, 0, 0]
        for nr, (integration_similarity_threshold, top_values_similarity_threshold, amount_top_answers) in enumerate(
                [(97, 97, 25), (95, 97, 15), (94, 97, 10), (92, 97, 7), (90, 97, 4), (88, 97, 4), (85, 97, 4)]):

            # if subs_nr:
            #     integration_similarity_threshold -= subs_nr * 1.5
            #     top_values_similarity_threshold -= subs_nr
            answers_to_embeddings = {k: v for k, v in answers_to_embeddings.items() if k[0] in value_counts}

            if len(answers_to_embeddings) < 3:
                continue
            
            # ic(row['column_name_normalized'])
                        
            unified_answers, not_found = calculate_cosine_similarities_between_answer_items(
                    answers_to_embeddings, value_counts, integration_similarity_threshold, 
                    row['column_name_normalized'], client, input_dir, 
                    nr)

            # subs_nr += 1

            if unified_answers:
                value_counts = update_vc(unified_answers, value_counts)    
                insert_nr += 1      
                row[f'unified answers {insert_nr}'] = value_counts if unified_answers else np.NaN
                row[f'replace_d unified answers {insert_nr}'] = dict(unified_answers) if unified_answers else np.NaN
                row[f'unified answers not found {insert_nr}'] = list(not_found) if not_found and unified_answers else np.NaN
        #input()
        # row[f'unified answers not clustered {nr}'] = not_clustered

    return row

def update_vc(unified_answers, value_counts):
    updated_vc = {}
    integrated = {}
    for answer_max, answer_to_integrate_l in unified_answers.items():
        # if answer_max == 'others/rest':
        #     perc_max = 0
        # else:
        perc_max = value_counts[answer_max]
        for answer in answer_to_integrate_l:
            perc = value_counts[answer]
            perc_max += perc
            integrated[answer] = answer_max
        updated_vc[answer_max] = perc_max
    updated_vc_rest = {k: v for k, v in value_counts.items() if k not in unified_answers and k not in integrated}
    inters = set(updated_vc.keys()).intersection(set(updated_vc_rest.keys()))
    if inters:
        ic(inters, updated_vc, updated_vc_rest)
        raise CustomError('intersection of keys in updated_vc and updated_vc_rest')
    updated_vc = {**updated_vc, **updated_vc_rest}
    updated_vc = pd.Series(updated_vc)
    # updated_vc.name = value_counts.name
    updated_vc = updated_vc.sort_values(ascending=False)
    
    #sanity check
    if sum(updated_vc) < sum(value_counts) - 0.1 or sum(updated_vc) > sum(value_counts) + 0.1:
        # ic(updated_vc, value_counts)
        ic(sum(updated_vc), sum(value_counts))
        warn('sum(updated_vc) != sum(value_counts)')
    
    #sanity check
    if sum(updated_vc) < 99.8 or sum(updated_vc) > 100.2:
        if sum(updated_vc) < sum(value_counts) - 0.1 or sum(updated_vc) > sum(value_counts) + 0.1:
            ic(sum(updated_vc), sum(value_counts))
            warn('sum(updated_vc) != 100')
    # ic(updated_vc)
    # ic(value_counts)
    return updated_vc

def negation_test(answer):
    negate_l = [' no', 'no ', ' non', 'non ', ' not', 'not ', ' any', 'any ',
            "don't", "doesn't", "didn't", "won't", "wouldn't", "couldn't", 
            "shouldn't", "can't", "can not", "cannot", "public", "outside", "yard", "plot", "elsewhere", "elswhere", "other", "compound"]
    answer_s = set(answer.split(' '))
    for negate in negate_l:
        if negate in answer_s:
            return True
    return False


def calculate_cosine_similarities_between_answer_items(answers, value_counts, similarity_threshold, col_name, client, input_dir, iter_nr):
    # Step 1a: Sort the items of 'answers'
    sorted_items = sorted(answers.items(), key=lambda x: value_counts.get(x[0][0], 0), reverse=True)
    answers = dict(sorted_items)
    
    try:
        # Calculate question embedding
        col_name_embedding = get_embedding((col_name, col_name), client, col_name, input_dir, model='text-embedding-ada-002')
    except Exception as e:
        ic(col_name)
        raise(e)
    # Step 1: Preprocessing and Data Structure Preparation
    all_embeddings = [col_name_embedding]
    answers_to_words = {}
    word_to_index = {col_name: 0}  
    # all_answer_embeddings = []
    all_answers = []

    for (answer, answer_norm), (answer_embedding, embeddings) in answers.items():
        answer_norm_words = answer_norm.split(' ')
        if '' in answer_norm_words:
            answer_norm_words.remove('')
            # raise ValueError('Empty word in answer_norm_words', answer_norm_words)
        if not answer_norm_words:
            answer_norm_words = [answer_norm]
        answers_to_words[answer] = answer_norm_words
        # all_answer_embeddings.append(answer_embedding)
        all_answers.append(answer)
        if answer not in word_to_index:
            word_to_index[answer] = len(all_embeddings)
            all_embeddings.append(answer_embedding)
        # ic(answer_norm_words)
        # embeddings_of_answer[answer] = pd.DataFrame(cosine_similarity([answer_embedding].extend(embeddings)), index=[answer].extend(answer_norm_words), columns=[answer].extend(answer_norm_words)) * 100
        for word, embedding in zip(answer_norm_words, embeddings):
            if word not in word_to_index:
                word_to_index[word] = len(all_embeddings)
                all_embeddings.append(embedding)
    words_l = list(word_to_index.keys())
    similarity_df = pd.DataFrame(cosine_similarity(all_embeddings), index=words_l, columns=words_l) * 100
    
    # similarity_df_answers = pd.DataFrame(cosine_similarity(all_answer_embeddings), index=all_answers, columns=all_answers) * 100

    # Step 4: Find Best Match for Each Answer Below the Threshold
    all_similarities = defaultdict(dict)
    for nr, (top_answer, answer_norm) in enumerate(answers):
        answers_to_compare_with = [k[0] for k, v in list(answers.items())[nr + 1:]]
        # Calculate similarity with top answers and find the best match
        for answer in answers_to_compare_with:
            if answer == top_answer:
                continue
            all_similarities[answer][top_answer] = retrieve_similarity(answer, top_answer, similarity_df, answers_to_words, col_name, 
                                                          False)

    # Step 5: Unify Answers
    found = {}
    found_s = set()
    final_clusters = defaultdict(list)
    # ic(all_similarities)
    for integration_candidate, sub_dict in all_similarities.items():
        #sort sub_dict by value[0]
        # ic(integration_candidate, len(sub_dict), sub_dict)
        sub_dict = {k: v for k, v in sorted(sub_dict.items(), key=lambda item: item[1][0], reverse=True)}
      
        best_match = next(iter(sub_dict))
        (avg_max_similarity, avg_max_similarity_weighted, similarity_to_answer, different_word_similarity, matched, longer_words, shorter_words) = sub_dict[best_match]
        
        local_similarity_threshold = similarity_threshold
        
        # Penalties
        # didnt match the highest priority word(s) (weighted with word - question similarity)
        if not matched:
            local_similarity_threshold += 1 * (10 - iter_nr)
        
        # Penalty if only substrings are matched (note they might have the same length regardless of name)
        # if len(longer_words) != len(shorter_words):
        #     local_similarity_threshold += 1 * (10 - iter_nr)
        
        # Adjust threshold based on percentage of answer
        try:
            perc = value_counts[integration_candidate]
        except Exception as e:
            print(integration_candidate)
            print(value_counts.head(200))
            print(integration_candidate in value_counts, '\n')
            ic(integration_candidate, value_counts, integration_candidate in value_counts, '\n')
            # ic('\n')
            raise(e)

        if perc < 0.05:
            local_similarity_threshold -= 10 * iter_nr        
        elif perc < 0.2:
            local_similarity_threshold -= 7 * iter_nr
        elif perc < 0.5:
            local_similarity_threshold -= 6 * iter_nr
        elif perc < 1:
            local_similarity_threshold -= 4 * iter_nr
        # elif perc < 2:
        #     local_similarity_threshold -= 0.5 * iter_nr
        elif perc > 20:
            local_similarity_threshold += 4 * (7 - iter_nr)
        elif perc > 15:
            local_similarity_threshold += 3 * (7 - iter_nr)
        elif perc > 10:
            local_similarity_threshold += 2 * (7 - iter_nr)
        elif perc > 7:
            local_similarity_threshold += 1.75 * (6 - iter_nr)
        elif perc > 4:
            local_similarity_threshold += 1 * (6 - iter_nr)
        elif perc > 2:
            local_similarity_threshold += 0.5 * (6 - iter_nr)
        elif perc > 1:
            local_similarity_threshold += 0.25 * (6 - iter_nr)
                            
        if avg_max_similarity > local_similarity_threshold:
            # ic("sub_dict: associated answer, (avg_max_similarity, avg_max_similarity_weighted, similarity(answer_integration_candidate, answer2), different_word_similarity, matched)")  
            # for k, v in sub_dict.items():
            #     ic(k, v)
            # ic("best_match", iter_nr, integration_candidate, best_match, local_similarity_threshold, perc, avg_max_similarity, avg_max_similarity_weighted, similarity_to_answer, different_word_similarity, matched, longer_words, shorter_words)

            if integration_candidate in found_s:
                ic('Should not happen', integration_candidate, best_match)
            if best_match in found:
                final_clusters[found[best_match]].append(integration_candidate)
                found[integration_candidate] = found[best_match]
            else:
                found[integration_candidate] = best_match
                final_clusters[best_match].append(integration_candidate)
    
    found = set(final_clusters.keys())
    found2 = set()
    for l in final_clusters.values():
        found2 = found2.union(set(l))    
    not_found = set([a[0] for a in answers]) - (found.union(found2))
    #ic(found, found2, not_found)
    # ic(value_counts.head(15))
    # ic(dict(sorted(final_clusters.items())))
    # ic(len(answers), len(found), len(found2), len(not_found), sum([len(found), len(found2), len(not_found)]))
    return final_clusters, not_found


def retrieve_similarity(answer_integration_candidate, answer2, similarity_df, answers_to_words, col_name, col_base):
    # Extract word lists for each answer
    words_l1 = answers_to_words[answer_integration_candidate].copy()
    words_l2 = answers_to_words[answer2].copy()
    if len(words_l1) == 0 or len(words_l2) == 0:
        ic(col_name, words_l1, words_l2)
        raise ValueError('Empty word list', words_l1, words_l2)
    
    # Calculate weights (similarity to col_name) for each word
    exponent = 3  # Adjust this value to control the scaling effect
    weights_l1 = {word: similarity_df.loc[word, col_name] ** exponent for word in words_l1}
    if col_base and col_base != col_name:
        weights_l2 = {word: similarity_df.loc[word, col_base] ** exponent for word in words_l2}
    else:
        weights_l2 = {word: similarity_df.loc[word, col_name] ** exponent for word in words_l2}

    # Calculate similarities for the rest of the words
    shorter_words, longer_words = (words_l2, words_l1) if len(words_l1) >= len(words_l2) else (words_l1, words_l2)
    shorter_weights, longer_weights = (weights_l2, weights_l1) if len(words_l1) >= len(words_l2) else (weights_l1, weights_l2)
    # ic(shorter_weights, longer_weights)

    matched = []
    # create 1-2 words in the longer answer that must be matched (based on the weights/ importance to the question)
    sorted_l_weights = [k for k, v in sorted(longer_weights.items(), key=lambda x: x[1], reverse=True)]
    if len(sorted_l_weights) > 2:
        must_match_words = sorted_l_weights[:2]
    else:
        must_match_words = sorted_l_weights[:1]
    
    # longer_words_orig = longer_words.copy() 
    matched_words = {}
    debugging_d = defaultdict(list)
    longer_words_b = longer_words.copy()
    similarity_l = []
    weight_l = []
    for word1 in shorter_words:
        best_sim = 0
        best_weight = 0
        best_match = None
        for word2 in longer_words_b:
            word_similarity = similarity_df.loc[word1, word2]
            weight = (shorter_weights[word1] + longer_weights[word2]) / 2
            debugging_d[word1].append((word2, word_similarity, weight, word_similarity * weight))
            # ic(word1, word2, shorter_weights.get(word1, 1), longer_weights.get(word2, 1))
            if word_similarity > best_sim:
                best_sim = word_similarity
                best_match = word2
                best_weight = weight
        #make sure the same word is not matched twice
        longer_words_b.remove(best_match)
        matched_words[word1] = best_match
        # if 'gas' in words_l1 or 'gas' in words_l2:
        #     ic(answer_integration_candidate, answer2, word1, best_match, best_sim, best_weight)
        if best_match in must_match_words:
            matched.append(best_match)
        similarity_l.append(best_sim)
        weight_l.append(best_weight)
        # total_weighted_similarity += best_sim * best_weight
        # total_weight += best_weight
    # ic(best_match, best_sim)

    # Calculate average maximum similarity for the rest of the words
    avg_max_similarity_weighted = (sum([s * w for s, w in zip(similarity_l, weight_l)]) / sum(weight_l)) if sum(weight_l) > 0 else 0
    
    # Incorporate a higher weighting of the similarities of non (near) identical words e.g. public well vs protected well -> more focus on public vs protected
    different_word_similarity = False
    different_words_similarities = [s for s in similarity_l if s < 99]
    if different_words_similarities and len(similarity_l) > len(different_words_similarities):
        different_word_similarity = sum(different_words_similarities) / len(different_words_similarities)
        avg_max_similarity = (avg_max_similarity_weighted + different_word_similarity) / 2
    else:
        avg_max_similarity = avg_max_similarity_weighted
    
    len_dif = len(longer_words) - len(shorter_words)
    if len_dif > 0:
        avg_max_similarity -= len_dif/len(longer_words) * 5
    
    highly_important_matches = [['flush', 'flushed', 'pipe', 'piped', 'tap', 'tapped', 'faucet'],
                                ['neighbour', 'neighbor', 'naighbor', 'someone'],
                                ['outside', 'yard', 'plot', 'compound'], 
                                ['dwelling', 'house', 'home', 'hut', 'residence', 'resid', 'residance', 'resid'],
                                ['imp', 'improved', 'improv', 'vip', 'vent', 'ventilated']]
    
    shorter_words_set = set(shorter_words)
    longer_words_set = set(longer_words)
    neg_s = set(['no', 'not', 'any', 'without', 'none', 'non'])
    
    condition_shorter = True if shorter_words_set & neg_s else False
    if not condition_shorter:
        condition_shorter = any([True for w in shorter_words if w[:2] == 'un'])
    condition_longer = True if shorter_words_set & neg_s else False
    if not condition_longer:
        condition_longer = any([True for w in shorter_words if w[:2] == 'un'])

    # lower similarity if negation is not matched
    if condition_shorter != condition_longer:
        avg_max_similarity -= 5
    
    #check important matches
    for nr, words_lx in enumerate(highly_important_matches):
        m = 1
        if nr == 1 or nr == 3:
            m = 2
        words_lx_set = set(words_lx)
        if words_lx_set & shorter_words_set and words_lx_set & longer_words_set:
            # avg_max_similarity += 2.5 * m
            pass
        elif not words_lx_set & shorter_words_set and not words_lx_set & longer_words_set:
            pass
        else:               
            avg_max_similarity -= 5 * m

    return avg_max_similarity, avg_max_similarity_weighted, similarity_df.loc[answer_integration_candidate, answer2], different_word_similarity, matched, longer_words, shorter_words

    # Adjust by similarity of complete answers
    # avg_max_similarity = (avg_max_similarity_weighted + similarity_df_answers.loc[answer_integration_candidate, answer2]) / 2

    # if 'community stand pipe' in answer_integration_candidate or 'community stand pipe' in answer2:
    # ic(answer_integration_candidate, answer2, avg_max_similarity, avg_max_similarity_weighted, 
    #    similarity_df_answers.loc[answer_integration_candidate, answer2], shorter_words, longer_words,
    #    shorter_weights, longer_weights)
    # if different_words_similarities and len(similarity_l) > len(different_words_similarities):
    #     ic(different_word_similarity, different_words_similarities)
    # sort debugging dict by similarity
    # debugging_d = {k: sorted(v, key=lambda x: x[1], reverse=True) for k, v in debugging_d.items()}
    # ic('debugging_d')
    # for k, v in debugging_d.items():
    #     ic(k, v)
    # if not matched:
    #     ic('not matched', must_match_words, '\n')
    # else:
    #     ic('matched', matched, must_match_words, '\n')
    
    
    #Penalties
    # didnt match the highest priority word(s) (weighted with word - question similarity)
    # if not matched:
    #     avg_max_similarity -= 10
    
    # # Penalty if only substrings are matched
    # if len(longer_words) > len(shorter_words):
    #     avg_max_similarity -= penalty_for_uneven_word_count
    
    # ic('final', avg_max_similarity, '\n')
    # Penalty if the highest similarity words of the longer answer to the column name are not matched

        
    # # Custom stuff
    # words_s1 = set(words_l1)
    # words_s2 = set(words_l2)
    # test_s = set(['outside', 'yard', 'plot', 'compound'])
    # len1, len2 = len(words_s1.intersection(test_s)), len(words_s2.intersection(test_s))
    # if len1 > 0 and len2 == 0 or len1 == 0 and len2 > 0:
    #     avg_max_similarity -= 10
        
    # test_s = set(['public'])
    # len1, len2 = len(words_s1.intersection(test_s)), len(words_s2.intersection(test_s))
    # if len1 > 0 and len2 == 0 or len1 == 0 and len2 > 0:
    #     avg_max_similarity -= 10
        
    # test_s = set(['neighbor', 'neighbour', 'naighbor'])
    # len1, len2 = len(words_s1.intersection(test_s)), len(words_s2.intersection(test_s))
    # if len1 > 0 and len2 == 0 or len1 == 0 and len2 > 0:
    #     avg_max_similarity -= 10
        
    # test_s = set(['no', 'non', 'not', 'any',
    #         "don't", "doesn't", "didn't", "won't", "wouldn't", "couldn't", 
    #         "shouldn't", "can't", "can not", "cannot", "without"])
    # len1, len2 = len(words_s1.intersection(test_s)), len(words_s2.intersection(test_s))
    # if (len1 > 0 and len2 == 0 and 'with' not in words_s1) or (len1 == 0 and len2 > 0 and 'with' not in words_s2):
    #     avg_max_similarity -= 10

    # test_s = set(['dwelling', 'house', 'home', 'hut', 'residence', 'resid', 'residance', 'resid.'])
    # len1, len2 = len(words_s1.intersection(test_s)), len(words_s2.intersection(test_s))
    # if len1 > 0 and len2 == 0 or len1 == 0 and len2 > 0:
    #     avg_max_similarity -= 10
        
    # #add small malus for different lengthes
    # if avg_max_similarity < 95 and len(shorter_words) != len(longer_words_orig) and len(shorter_words) == 1:
    #     avg_max_similarity -= 5
    #     #add extra malus for different length (e.g. rainwater matches with (bottled) water etc.)
    #     if len(shorter_words[0]) > len(matched_words[shorter_words[0]]):
    #         avg_max_similarity -= len(shorter_words[0]) - len(matched_words[shorter_words[0]]) * 2
        
    # if 'gas' in words_l1 or 'gas' in words_l2:
    #     ic(answer_integration_candidate, answer2, words_l1, words_l2, avg_max_similarity, '\n')
        

def negation_test(answer):
    negate_l = [' no', 'no ', ' non', 'non ', ' not', 'not ', ' any', 'any ',
            "don't", "doesn't", "didn't", "won't", "wouldn't", "couldn't", 
            "shouldn't", "can't", "can not", "cannot", "public", "outside", "yard", "plot", "elsewhere", "elswhere", "other", "compound"]
    answer_s = set(answer.split(' '))
    for negate in negate_l:
        if negate in answer_s:
            return True
    return False




#############################Preprocessing specific functions#############################################

def preprocessing_loading_wrapper(cols, file_paths, dataset_type, folder_path):
    all_data, meta_data = load_data(file_paths, dataset_type, folder_path, col_labels=cols, create_GEID=False, create_DHSID=True, 
                                max_workers=4, create_version_nr=False, load_keys=False)
    df = combine_survey_dfs(all_data)[0]
    return df


def preprocessing_loading_wrapper_wrapper(file_paths, dataset_type, folder_path, cols_to_load_simultaneously, preprocessing_answers_df):
    cols_l_in = preprocessing_answers_df["column_name"].tolist()
    cols_l = [cols_l_in[i:i + cols_to_load_simultaneously] for i in range(0, len(cols_l_in), cols_to_load_simultaneously)]
    # create meta data first
    all_data, meta_data = load_data(file_paths, dataset_type, folder_path, col_labels=['country code and phase'], create_GEID=True, create_DHSID=True, 
                                max_workers=47, create_version_nr=True, load_keys=False)
    meta_df = combine_survey_dfs(all_data)[0]

    partial_preprocessing_loading_wrapper = partial(preprocessing_loading_wrapper, file_paths=file_paths, dataset_type=dataset_type, folder_path=folder_path)
    with ProcessPoolExecutor(max_workers=len(cols_l[0])) as executor:
        results = list(tqdm(executor.map(partial_preprocessing_loading_wrapper, cols_l), total=len(cols_l)))
    
    combined_df = meta_df
    for nr, df in enumerate(results):
        if df['DHSID + HHID'].nunique() != len(df):
            ic(nr, len(df), len(combined_df))
            ic(df['DHSID + HHID'].nunique(), combined_df['DHSID + HHID'].nunique())
            # ic non unique rows
            df_t = df[df.duplicated(subset=['DHSID + HHID'], keep=False)]
            asd = df_t['DHSID'].str[:8]
            ic(asd.nunique(), asd.unique())
            raise ValueError('not unique rows')
        
        assert len(combined_df) == len(df)
        df = df.drop(columns=[c for c in df.columns if c in combined_df.columns and c != 'DHSID + HHID'])
        combined_df = combined_df.merge(df, on='DHSID + HHID', how='outer')
    return combined_df


def create_replace_d_from_v_list_keys(replace_d_in):
    replace_d = {}
    for col_n, replace_d_col in replace_d_in.items():
        replace_d[col_n] = {}
        for key, values_l in replace_d_col.items():
            for v in values_l:
                if not v and v != 0:
                    ic(v, col_n, key, values_l)
                # if v is not '':
                else:
                    replace_d[col_n][v] = key
                    if isinstance(v, (int, float, complex)):
                        replace_d[col_n][str(v)] = key
                
    return replace_d


def unify_answers_after_preprocessing(df, preprocessing_df, load_dont_know_as_nan_perc=1, summarize_low_perc_to_other_perc=0.5):
    # Numeric values as keys
    replace_d = {}
    for decision, col, replace_d_cell in zip(preprocessing_df['numeric values as keys decision'].str[:4], 
                                             preprocessing_df['column_name'], preprocessing_df['replace_d numeric values as keys']):
        if decision == 'True' or decision is True:
            try:
                dict_obj = ast.literal_eval(replace_d_cell)
            except Exception as e:
                ic(col, replace_d_cell, e)
                raise(e)
            replace_d[col] = dict_obj
    ic('numeric values as keys', replace_d)
    df = df.replace(replace_d)
    
    # Corrected values
    replace_d = {}
    for col, replace_d_cell in zip(preprocessing_df['column_name'], preprocessing_df['replace_d corrected_answers']):
        if pd.notna(replace_d_cell):
            try:
                dict_obj = ast.literal_eval(replace_d_cell)
            except Exception as e:
                ic(col, replace_d_cell, e)
                raise(e)
            replace_d[col] = dict_obj

    ic('corrected values', replace_d)
    df = df.replace(replace_d)
    
    replace_d = {'month of interview': {'january': 1.0, 'february': 2.0, 'march': 3.0, 'april': 4.0, 'may': 5.0,
                                        'june': 6.0, 'july': 7.0, 'august': 8.0, 'september': 9.0, 'october': 10.0, 'november': 11.0, 'december': 12.0, 
                                        '1': 1.0, '2': 2.0, '3': 3.0, '4': 4.0, '5': 5.0, '6': 6.0, '7': 7.0, '8': 8.0, '9': 9.0}}
    ic('month of interview', replace_d)
    df = df.replace(replace_d)
    
    # Unified answers
    for replace_step, replace_col in enumerate(['replace_d unified answers 1', 'replace_d unified answers 2', 
                                                'replace_d unified answers 3', 'replace_d unified answers 4', 
                                                'replace_d unified answers 5', 'replace_d unified answers 6']):
        replace_d = {}
        for decision, col, replace_d_cell in zip(preprocessing_df['replace_d to use'], 
                                                 preprocessing_df['column_name'], 
                                                 preprocessing_df[replace_col]):
            if pd.notna(decision):
                decision = int(decision)
                if decision >= replace_step + 1:
                    if pd.notna(replace_d_cell):
                        try:
                            dict_obj = ast.literal_eval(replace_d_cell)
                        except Exception as e:
                            ic(col, replace_col, replace_step, replace_d_cell, e)
                            raise(e)
                        # ic(dict_obj, '\n')
                        replace_d[col] = dict_obj
        if replace_d:
            ic(replace_step, replace_d)
        replace_d = create_replace_d_from_v_list_keys(replace_d)
        df = df.replace(replace_d)
    
    replace_d = {}
    for col in df.columns:
        vc = df[col].value_counts(dropna=False, normalize=True) * 100
        if "don't know" in vc:
            if vc["don't know"] < load_dont_know_as_nan_perc:
                replace_d[col] = {"don't know": np.NaN}
        # #drop na row
        # if vc.hasnans:
        #     vc = vc.drop(labels=[np.nan])
        # if vc.max() >= 10 and len(vc) >:
        #     vc_below_perc = vc[vc < summarize_low_perc_to_other_perc]
        #     if len(vc_below_perc) > 1:
        #         ic('dropping stuff', col)
        #         replace_d[col] = {b_perc: 'other' for b_perc in vc_below_perc.index if pd.notna(b_perc)} 
    
    replace_d_ic = {k: {k2: (v2 if pd.notna(v2) else 'NA') for k2, v2 in v.items()} for k, v in replace_d.items()}
    ic('replace_d dont know', replace_d_ic)
    df = df.replace(replace_d)
    
    # Custom replace dict for ':' bug
    replace_d = {
                 'water usually treated by: other': {'yes': ['yes: other']},
                 'water usually treated by: let it stand and settle': {'yes': ['yes: let it stand and settle']},
                 'water usually treated by: solar disinfection': {'yes': ['yes: solar disinfection']}, 
                 'water usually treated by: use water filter': {'yes': ['yes: use water filter']},
                 'water usually treated by: strain through a cloth': {'yes': ['yes: strain through a cloth']},
                 'hectares of agricultural land (1 decimal)': {38.0: ['95 or more acres']},
                 'has television': {'yes': [1.0], 'no': [0.0]},
                 'time to get to water source (minutes)': {0.0: ['on premises'], 1.0: ['less than  minute'], 900.0: ['one day or longer'], 720: ['more than 12 hours']},
                 'year of interview': {1999: [99.0], 1998: [98.0], 1997: [97.0], 1996: [96.0], 1995: [95.0], 1994: [94.0], 1993: [93.0]}
                }
    replace_d = create_replace_d_from_v_list_keys(replace_d)
    ic('custom replace_d', replace_d)
    df = df.replace(replace_d)
    return df
    
    
def preprocessing_loading_wrapper_unified_columns(group, file_paths, dataset_type, folder_path, max_workers=1):
    cols = group['column_name'].tolist()
    all_data, meta_data = load_data(file_paths, dataset_type, folder_path, col_labels=cols, create_GEID=False, create_DHSID=True, 
                                max_workers=max_workers, create_version_nr=False, load_keys=False)
    df = combine_survey_dfs(all_data)[0]
       
    # ic('next group', cols, df.columns)
    # Do some preprocessing
    for col in cols:
        row = group[group['column_name'] == col]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        # ic(col)
        # ic(type(row))
        # ic(row['available_data [%]'])
        
        #Replacing numeric values and corrected_answers before unifying columns (not guaranteed to have the same replace_dicts)
        replace_d = {}
        decision = str(row['numeric values as keys decision'])[:4]
        # ic('hi')
        # ic(decision, type(decision))
        # decision = row['numeric values as keys decision'][:4]
        replace_d_cell = row['replace_d numeric values as keys']
        if decision == 'True':
            try:
                dict_obj = ast.literal_eval(replace_d_cell)
            except Exception as e:
                ic(col, replace_d_cell, e)
                raise(e)
            replace_d[col] = dict_obj
        # ic('numeric values as keys', col, replace_d)
        df = df.replace(replace_d)
        
        # Corrected values
        replace_d = {}
        replace_d_cell = row['replace_d corrected_answers']
        if pd.notna(replace_d_cell):
            try:
                dict_obj = ast.literal_eval(replace_d_cell)
            except Exception as e:
                ic(col, replace_d_cell, e)
                raise(e)
            replace_d[col] = dict_obj

        # ic('corrected values', replace_d)
        df = df.replace(replace_d)
        
        # Adjust the answers so they match the answers of the column to integrate in
        replace_d = {}
        replace_d_cell = row['answer mapping for integration']
        if pd.notna(replace_d_cell):
            try:
                dict_obj = ast.literal_eval(replace_d_cell)
            except Exception as e:
                ic(col, replace_d_cell, e)
                raise(e)
            replace_d[col] = dict_obj

        # ic('answer mapping for integration', replace_d)
        df = df.replace(replace_d)
        
    # Unify columns
    first_col = cols[0]
    rest_cols = cols[1:]
    
    assert group[group['column_name'] == first_col].iloc[0]['available_data [%]'] == group['available_data [%]'].max()
    
    for rcol in rest_cols:
        df[first_col] = df[first_col].fillna(df[rcol])
    
    df = df[[first_col, 'DHSID + HHID']]
        
    return df, cols

    
def preprocessing_unified_columns(unify_columns_df, file_paths, dataset_type, folder_path, cut_perc):
    # Get relevant groups
    group_l = unify_columns_df.groupby('similar_columns nr')
    ic(len(group_l))
    
    # Sort each group in group_l by 'available_data [%]' in descending order
    new_group_l = [group.sort_values(by=['available_data [%]'], ascending=False) for _, group in group_l]

    # For each group, append rows where 'overall indicator short' is True
    # Note top_row has no overall indicator short (cause it is the one into which the other cols are integrated)
    newx_group_l = []
    for group in new_group_l:
        top_row = group.iloc[[0]]
        rest_group_rows = group.iloc[1:].copy()

        # change to string and strip - only retrieve cols that shall be unified
        rest_group_rows.loc[:, 'overall indicator short'] = rest_group_rows['overall indicator short'].astype(str).str.strip().str.upper()
        rest_group_rows = rest_group_rows[rest_group_rows['overall indicator short'] == 'TRUE']
        
        if len(rest_group_rows) > 0:
            group = pd.concat([top_row, rest_group_rows], axis=0)
            newx_group_l.append(group)
    
    new_group_l = [group for group in newx_group_l if group['available_data [%]'].sum() >= cut_perc]
        
    ic(len(new_group_l))
    
    partial_preprocessing_loading_wrapper = partial(preprocessing_loading_wrapper_unified_columns, file_paths=file_paths, dataset_type=dataset_type, folder_path=folder_path)
    with ProcessPoolExecutor(max_workers=47) as executor:
        results = list(tqdm(executor.map(partial_preprocessing_loading_wrapper, new_group_l), total=len(new_group_l)))
    
    all_cols = [item for sublist in results for item in sublist[1]]
    results = [r[0] for r in results]
    combined_df = results[0]
    ic(len(results), combined_df, all_cols)
    for nr, df in enumerate(results[1:]):
        if df['DHSID + HHID'].nunique() != len(df):
            ic(nr, len(df), len(combined_df))
            ic(df['DHSID + HHID'].nunique(), combined_df['DHSID + HHID'].nunique())
            # ic non unique rows
            df_t = df[df.duplicated(subset=['DHSID + HHID'], keep=False)]
            asd = df_t['DHSID'].str[:8]
            ic(asd.nunique(), asd.unique())
            raise ValueError('not unique rows')
        
        assert len(combined_df) == len(df)
        df = df.drop(columns=[c for c in df.columns if c in combined_df.columns and c != 'DHSID + HHID'])
        combined_df = combined_df.merge(df, on='DHSID + HHID', how='outer')
    return combined_df, set(all_cols)
    

def split_label_df(df_in, sub_df_ind, value_col, force_split_col2, split_col_n, img_path, excl_outlier, force_test_ind, force_into, since_year=2012, split_amount=6, excl_outlier_std=1, assign_test=False):
    """
    Split the label DataFrame according to an x-cross evaluation.

    Parameters
    ----------
    df_in : pandas.DataFrame
        The input label DataFrame to split.
    mode : str
        The mode to use for splitting and labeling the DataFrame. Valid options include:
            - 'split: random': Randomly split the DataFrame into `split_amount` splits.
            - 'split: without <2015': Split the DataFrame into `split_amount` splits, but exclude any rows with a DHSYEAR less than 2015.
            - 'split: country year': Split the DataFrame by unique values of the `GEID` column, ensuring that each split does not contain multiple surveys from the same country.
            - 'split: out of country': Split the DataFrame by unique values of the first two characters of the `GEID` column.
    sub_df_ind : str or bool, optional
        If 'rural', only include rows where the `URBAN_RURA` column is 'R'. If 'urban', only include rows where the `URBAN_RURA` column is 'U'. If 'all' (default) or False, include all rows.
    split_amount : int, optional
        The number of splits to create (default 6).
    last_split_as_test : bool, optional
        If True (default), the last split will be labeled as 'test' instead of 'split N'.
    force_test_ind : str or bool, optional
        If a string is provided, any split containing a row with a value of `force_test_ind` in the column used for splitting will be labeled as 'test'.
         E.g 'MZ' ('split: out of country') or 'MZGE7AFL' ('split: out of country') can be provided to put Mozambique (2018) into the test set.
         If False (default), this behavior is disabled.
    img_path : bool, str
        If the path to the geotiff folder is passed omits rows where there are no geotiffs available else False

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with an additional column indicating the split based on the chosen mode.
    """
    df_in = df_in.copy()
    df = prepare_df(df_in, sub_df_ind, img_path, excl_outlier, excl_outlier_std, split_col_n, value_col, since_year=since_year)
#     print('1.1', df[split_col_n].value_counts())

    if 'split: random' in split_col_n:
        df = df.sample(frac=1)
        for i in range(0, split_amount):
            df[split_col_n].iloc[int(len(df)/split_amount) * (i): int(len(df)/split_amount) * (i+1)] = 'split ' + str(i)

        if assign_test:
            scores = hu.statistical_weighted_test_set(df, split_col_n, value_col)
            df.loc[df[split_col_n] == scores[0][0], split_col_n] = 'test'
        df_in = pd.merge(df_in, df[["DHSID", split_col_n]], how="outer", on=['DHSID'])
    else:
        column = 'split col'
        if 'year' in split_col_n:
            df[column] = df[force_split_col2] + df['DHSYEAR'].astype(int).astype(str)
        else:
            df[column] = df[force_split_col2]

        # Shuffle the dataframe
        df_shuffled = df.sample(frac=1).reset_index(drop=True)
        # Get the unique values of the categorical column
        unique_values = df_shuffled[column].unique()
        # Initialize a dictionary to store the rows for each unique value
        value_rows = {value: df_shuffled[df_shuffled[column] == value] for value in unique_values}
        #print(value_rows)
        # Sort the unique values by decreasing order of count
        sorted_values = {k: v for k, v in sorted(value_rows.items(), key=lambda item: len(item[1]), reverse=True)}
        sorted_dfs = [pd.DataFrame(columns=df.columns) for i in range(split_amount)]
        look_up_d = {i: {} for i in range(split_amount)}
        look_up_d_new = {}
        for unique_value, sub_df in sorted_values.items():
            if sub_df[force_split_col2].iloc[0] not in look_up_d:
                look_up_d_new[sub_df[force_split_col2].iloc[0]] = {}
            if 'year' in split_col_n:
                #make sure that it will be sorted into a split where the same country is not in it right now
                put_into_split = False
                for nr, sdf in enumerate(sorted_dfs):
                    if nr not in look_up_d_new[sub_df[force_split_col2].iloc[0]]:
                        look_up_d_new[sub_df[force_split_col2].iloc[0]][nr] = set([])
                    if sub_df['adm0_name'].iloc[0] not in look_up_d_new[sub_df[force_split_col2].iloc[0]][nr]:
                        sorted_dfs[nr] = pd.concat([sdf, sub_df])
                        look_up_d_new[sub_df[force_split_col2].iloc[0]][nr].add(sub_df['adm0_name'].iloc[0])
                        put_into_split = True
                        break
                if not put_into_split:
                    #we exceeded the amount of splits
                    sub_df2 = pd.DataFrame(df[column].value_counts()).reset_index()
                    sub_df2.columns = [column, 'amount']
                    print('value counts\n', sub_df2)
                    print(split_col_n)
                    # print(look_up_d[sub_df['adm0_name'].iloc[0]])
                    print(f"To many surveys for {sub_df[force_split_col2].iloc[0]} cannot sort in without putting multiple surveys of that country into one split")
                    #sort the following by the amount of unique values
                    print(df.groupby(force_split_col2)[column].nunique().sort_values())
                    # print(df.groupby(force_split_col2)[column].unique())
                    print(df[df[force_split_col2] == sub_df[force_split_col2].iloc[0]][['DHSYEAR', 'GEID', force_split_col2, 'adm0_name']].value_counts())
                    raise NotImplementedError("Possibly just raise the amount of splits")
            else:
                #concat the sub_df of unique_value to the shortest df
                sorted_dfs[0] = pd.concat([sorted_dfs[0], sub_df])
            #sort so that always the df with the least amount of rows is on pos 0
            sorted_dfs = sorted(sorted_dfs, key=lambda x: len(x))

        #stick everything together
        fdf = pd.concat([sdf for sdf in sorted_dfs])
        df_in = pd.merge(df_in, fdf[["DHSID", split_col_n]], how="outer", on=['DHSID'])

        if assign_test:
            #assign a test split to either Mozambique or to split 0
            if force_test_ind:
                if isinstance(force_test_ind, str):
                    if force_into:
                        test_split_n = df_in[df_in['adm0_name'] == force_test_ind][split_col_n].iloc[0]
                    else:
                        for test_split_n in df_in[split_col_n].unique():
                            if force_test_ind not in df_in[df_in[split_col_n] == test_split_n]['adm0_name']:
                                break
                        else:
                            raise ValueError(f'This should not happen force_test_ind is {force_test_ind}')
                elif isinstance(force_test_ind, tuple):
                    if force_into:
                        test_split_n = df_in[(df_in['adm0_name'] == force_test_ind[0]) & (df_in['DHSYEAR'] == force_test_ind[1])][split_col_n].iloc[0]
                    else:
                        for test_split_n in df_in[split_col_n].unique():
                            #make sure force_test_ind (adm0_name) is not in the test split (every Moz survey should be in the train split)
                            if force_test_ind[0] not in df_in[df_in[split_col_n] == test_split_n]['adm0_name']:
                                break
                        else:
                            raise ValueError(f'This should not happen force_test_ind is {force_test_ind}')
                else:
                    raise ValueError(f'This should not happen force_test_ind is {force_test_ind}')
            else:
                scores = hu.statistical_weighted_test_set(df, split_col_n, value_col)
                test_split_n = scores[0][0]
            df_in.loc[df_in[split_col_n] == test_split_n, split_col_n] = 'test'
    return df_in


def prepare_df(df_in, sub_df_ind, img_path, excl_outlier, excl_outlier_std, split_col_n, value_col=False, since_year=2012):
    df = df_in.copy()
    df[split_col_n] = np.NaN
    if since_year:
        df = df[df['DHSYEAR'] >= since_year]
    #restrict to specified area type
    if sub_df_ind == 'rural':
        df = df[df['URBAN_RURA'] == 'R']
    elif sub_df_ind == 'urban':
        df = df[df['URBAN_RURA'] == 'U']
    elif sub_df_ind is False or sub_df_ind == 'all':
        pass
    else:
        raise NotImplementedError()
    #reducing to available images
    if img_path:
        available_files = hu.files_in_folder(img_path, return_pathes=False)
        # print(available_files)
        # print(df['TIF_name'])
        df["TIF_name"] = df['TIF_name'] + '.tif'
        df["TIF_name"] = df['TIF_name'].apply(lambda x: x if x in available_files else np.NaN)
        df = df[df["TIF_name"].notna()]
        if len(df) == 0:
            print(available_files[:5])
            print(df['TIF_name'].head())
            raise ValueError('found no suiting images')
    if excl_outlier:
        #excluding outlier surveys (e.g. Egypt, South-Africa)
        df = df[~df['GEID'].isin(drop_survey)]
        if excl_outlier == 'std':
            mean = df[value_col].mean()
            std = df[value_col].std()
            print(split_col_n, value_col, 'mean:', mean, 'STD', std)
            for cy in df["GEID"].unique():
                sub_df = df[df["GEID"] == cy]
                sub_mean = sub_df[value_col].mean()
                # print('sub_mean', sub_mean, 'vs', mean, 'std', std, 'exclstd', excl_outlier_std)
                if sub_mean > mean + excl_outlier_std * std or sub_mean < mean - excl_outlier_std * std:
                    print('dropping', cy, 'amount:', len(sub_df), 'mean:', sub_mean, 'vs', mean)
                    vis.standard_hist_from_df(sub_df[value_col], projects_p + '/imgs/excl_surveys/', cy, sub_df['adm0_name'].iloc[0] + '_' + str(int(sub_df['DHSYEAR'].iloc[0])))
                    #iteratively removing surveys with high deviation of parent distribution (measured on mean)
                    df = df[df["GEID"] != cy]
    df = df[(df[value_col] >= df[value_col].mean() - 3.5 * df[value_col].std()) & (df[value_col] <= df[value_col].mean() + 3.5 * df[value_col].std())]
    return df


def visualize_splits(df, value_col, split_col_n, projects_p):
    print('\n', f"{split_col_n} size={len(df[value_col])}, mean={df[value_col].mean():.2f}, std={df[value_col].std():.2f}, min={df[value_col].min():.2f}, max={df[value_col].max():.2f} (Whole ds)")
    vis.standard_hist_from_df(df[value_col], projects_p + '/imgs/splits/', split_col_n, title_in='Whole DS', minv=-3.5, maxv=3.5)
    sub_df = df[df[split_col_n].notna()]
    vis.standard_hist_from_df(sub_df[value_col], projects_p + '/imgs/splits/', split_col_n, title_in='WO excluded', minv=-3.5, maxv=3.5)
    print(split_col_n, f'size={len(sub_df[value_col])}, mean={sub_df[value_col].mean():.2f}, std={sub_df[value_col].std():.2f}, min={sub_df[value_col].min():.2f}, max={sub_df[value_col].max():.2f} (Whole ds without excluded areas)')
    means = []
    stds = []
    sizes = []
    for s in df[split_col_n].unique():
        split_data = df[df[split_col_n] == s][value_col]
        #test for isna with possible str input
        if pd.isnull(s):
            split_data = df[df[split_col_n].isna()][value_col]
            s = '                      NAN'
        else:
            means.append(split_data.mean())
            stds.append(split_data.std())
            sizes.append(len(split_data))
        # if np.isnan(s):
        #     split_data = df[df[split_col_n].isna()][value_col]
        split_col_n_p = split_col_n.replace(' ', '_')
        print(f"{split_col_n} {s}: size={len(split_data)}, mean={split_data.mean():.2f}, std={split_data.std():.2f}, min={split_data.min():.2f}, max={split_data.max():.2f}")
        vis.standard_hist_from_df(sub_df[value_col], projects_p + '/imgs/splits/', split_col_n_p, title_in=s, minv=-3.5, maxv=3.5)
    print(f"{split_col_n} std of means: {np.std(means):.2f}, std of stds: {np.std(stds):.2f}, std of sizes {np.std(sizes):.2f}")


#write a function that statistically compares the different splits
def stats_of_splits(df, value_col, split_col_n, force_split_col, split_dict):
    split_dict[split_col_n] = {}
    means = []
    stds = []
    sizes = []
    for v in ['mean', 'std', 'size', 'skew', 'kurtosis']:
        for spl in list(df[split_col_n].unique()) + ['whole ds', 'without excluded areas and imgs']:
            #how to break here if spl is not a number?
            if pd.isnull(spl):
                continue
            if spl == 'whole ds':
                sub_df = df[value_col]
            elif spl == 'without excluded areas and imgs':
                sub_df = df[df[split_col_n].notna()][value_col]
            else:
                sub_df = df[df[split_col_n] == spl][value_col]
            n = spl + ' ' + v
            if v == 'mean':
                split_dict[split_col_n][n] = sub_df.mean()
                if spl != 'whole ds' and spl != 'without excluded areas and imgs':
                    means.append(sub_df.mean())
            elif v == 'std':
                split_dict[split_col_n][n] = sub_df.std()
                if spl != 'whole ds' and spl != 'without excluded areas and imgs':
                    stds.append(sub_df.std())
            elif v == 'size':
                split_dict[split_col_n][n] = int(len(sub_df))
                if spl != 'whole ds' and spl != 'without excluded areas and imgs':
                    sizes.append(int(len(sub_df)))
            elif v == 'skew':
                split_dict[split_col_n][n] = sub_df.skew()
            elif v == 'kurtosis':
                split_dict[split_col_n][n] = sub_df.kurtosis()
    split_dict[split_col_n]['std of means'] = np.std(means)
    split_dict[split_col_n]['std of stds'] = np.std(stds)
    split_dict[split_col_n]['std of sizes'] = np.std(sizes)
    # stats_df = pd.DataFrame(split_dict)
    return split_dict


def split_df_accounting_for_mean(df_in, sub_df_ind, value_col, force_split_col, split_col_n, img_path, excl_outlier, force_test_ind, force_into, since_year, n_splits=6, excl_outlier_std=1, assign_test=False):
    df = df_in.copy()
    df = df.dropna(subset=[value_col])
    # print('1', 'adm0_name' in df.columns)
    # print('split_col_n', split_col_n)
    # print('force_split_col', force_split_col)   
    # print(value_col)
    # print(len(df[value_col].dropna()))
    
    split_columns = [force_split_col]
    if 'year' in split_col_n:
        split_columns.append('DHSYEAR')
    if 'adm0_name' not in split_columns:
        split_columns.append('adm0_name')
    
    df = df.dropna(subset=split_columns)
    
    data = prepare_df(df, sub_df_ind, img_path, excl_outlier, excl_outlier_std, split_col_n, value_col=value_col, since_year=since_year)
    parent_mean, parent_std = data[value_col].agg(['mean', 'std'])
    # print('2', 'adm0_name' in df.columns)
    
    # print(split_columns)
    # print(data.columns)
    # print(len(data))
    # for split_col in split_columns:
    #     print('T', split_col in data.columns)
    stats_df = data.groupby(split_columns)[value_col].agg(['mean', 'count']).reset_index()
    # print(len(stats_df))
    stats_df = stats_df[stats_df['mean'].notna()]
    # print(len(stats_df))
    stats_df['mean_diff_w'] = (stats_df['mean'] - parent_mean) * stats_df['count']/stats_df['count'].mean()
    stats_df['mean_abs_w'] = np.abs(stats_df['mean_diff_w'])
    #create a balanced weighting from 'mean_abs_w' and 'count'
    stats_df['sort_score'] = stats_df['mean_abs_w']/stats_df['mean_abs_w'].std() + (stats_df['count'] - stats_df['count'].mean()) / stats_df['count'].std()
    stats_df = stats_df.sort_values(['sort_score'], ascending=[False])
    # print('statsdf', stats_df)
    splits = [[] for _ in range(n_splits)]

    #check if other years of the same are already have been added to the split
    look_up_d = {i: {} for i in range(n_splits)}
    for k, (j, row1) in enumerate(stats_df.iterrows()):
        best_split = None
        min_score = float('inf')
        str_l = []
        for nr, split in enumerate(splits):
            if row1['adm0_name'] not in look_up_d[nr]:
                look_up_d[nr][row1['adm0_name']] = set([])

            temp_split = split + [row1]
            if split:
                try:
                    combined_mean_diff = np.nanmean([row['mean_diff_w'] for row in split])
                except RuntimeWarning as e:
                    print(e)
                    combined_mean_diff = 0
            else:
                combined_mean_diff = 0
            try:
                combined_mean_diff_new = np.nanmean([row['mean_diff_w'] for row in temp_split])
            #also print Warning if the mean is nan
            except RuntimeWarning as e:
                print(e)
                combined_mean_diff_new = 0

            #add penalty if the mean goes into the wrong direction
            if len(split) == 0:
                score = 0
            #add penalty if the mean goes into the wrong direction
            elif combined_mean_diff <= 0 <= row1['mean_diff_w'] or row1['mean_diff_w'] <= 0 <= combined_mean_diff:
                #if the mean switches from negative to positive, add the difference as a penalty else use it as a weighting to force the furthest away to have a high chance to be added
                if combined_mean_diff <= 0 <= combined_mean_diff_new or combined_mean_diff_new <= 0 <= combined_mean_diff:
                    score = abs(combined_mean_diff_new)
                else:
                    score = -abs(combined_mean_diff_new)
            else:
                #if it is pointing into the wrong direction, add it as a penalty though
                score = abs(combined_mean_diff_new) + abs(combined_mean_diff_new)

            # Calculate the penalty based on the difference of summed counts between the splits
            counts = [sum([r['count'] for r in s] + [row1['count']]) for s in splits]
            count_local = sum([r['count'] for r in split]) + row1['count']
            count_penalty = 1 * max(parent_std, abs(combined_mean_diff_new)) * (count_local - np.min(counts))/count_local * (len(stats_df)/2 + k/2)/len(stats_df)
            score += count_penalty
            str_l.append([nr, count_local, score, combined_mean_diff, combined_mean_diff_new, count_penalty, row1['mean_diff_w'], row1['count']])

            if score < min_score and ('year' not in split_col_n or row1[force_split_col] not in look_up_d[nr][row1['adm0_name']]):
                min_score = score
                best_split = nr
        # print('appending', 'to', best_split)
        if best_split is None:
            print(split_col_n)
            #sort the following by the amount of unique values
            # print(data.groupby(force_split_col)[force_split_col].nunique().sort_values())
            print(df[df[force_split_col] == row1[force_split_col]][['DHSYEAR', 'GEID', force_split_col, 'adm0_name']].value_counts())
            print(stats_df[stats_df[force_split_col] == row1[force_split_col]][split_columns].value_counts())
            print(f"To many surveys for {row1[force_split_col]} cannot sort in without putting multiple surveys of that country into one split")
            raise NotImplementedError("Possibly just raise the amount of splits")

        # if str_l[best_split][1] == max(c[1] for c in str_l) and k >= 2:
        # print(best_split)
        # for str_gen in str_l:
        #     print(f"k {k} split {str_gen[0]} count {str_gen[1]} score {str_gen[2]} combined_mean_diff {str_gen[3]} combined_mean_diff_new {str_gen[4]} count_penalty {str_gen[5]} row mean diff w {str_gen[6]} row count {str_gen[7]}")
        splits[best_split].append(row1)
        look_up_d[best_split][row1['adm0_name']].add(row1[force_split_col])
        # print('hi')
    # print('splits', splits)
    # Create a dictionary to map group keys (tuple of values from groupby_cols) to their split assignments
    split_assignments = {tuple(row[split_columns]): i for i, country_group in enumerate(splits) for row in country_group}
    print('spl', split_assignments)
    # Update the 'Split' column in the input data with the split assignments
    data[split_col_n] = data[split_columns].apply(lambda x: f'split {split_assignments[tuple(x)]}', axis=1)

    if assign_test:
        #assign a test split to either Mozambique or to split 0
        if force_test_ind:
            if isinstance(force_test_ind, str):
                if force_into:
                    test_split_n = data[data['adm0_name'] == force_test_ind][split_col_n].iloc[0]
                else:
                    for test_split_n in data[split_col_n].unique():
                        if force_test_ind not in data[data[split_col_n] == test_split_n]['adm0_name']:
                            break
                    else:
                        raise ValueError(f'This should not happen force_test_ind is {force_test_ind}')
            elif isinstance(force_test_ind, tuple):
                if force_into:
                    test_split_n = data[(data['adm0_name'] == force_test_ind[0]) & (data['DHSYEAR'] == force_test_ind[1])][split_col_n].iloc[0]
                else:
                    for test_split_n in data[split_col_n].unique():
                        #make sure force_test_ind (adm0_name) is not in the test split (every Moz survey should be in the train split)
                        if force_test_ind[0] not in data[data[split_col_n] == test_split_n]['adm0_name']:
                            break
                    else:
                        raise ValueError(f'This should not happen force_test_ind is {force_test_ind}')
            else:
                raise ValueError(f'This should not happen force_test_ind is {force_test_ind}')
        else:
            #create mean and std weighted scores
            scores = hu.statistical_weighted_test_set(df, split_col_n, value_col)
            test_split_n = scores[0][0]
        data.loc[data[split_col_n] == test_split_n, split_col_n] = 'test'
    print('1', len(data), len(df_in))
    ldfin = len(df_in)
    df_in = pd.merge(df_in, data[["DHSID", split_col_n]], how="left", on='DHSID')
    print(len(df_in))
    assert len(df_in) == ldfin
    return df_in
        
#Known issues:
# - to_csv does alter encoding so that some chars are missrepresented (needs to be fixed in .csv to correctly load from preprocessing)
# - 'datatype similarity' is bugged (rework based on string [%] and numeric [%]!? Check if they are updated!)
# - 'replace_d corrected_answers' is bugged and does not include some strings in minor? columns?
#   - workaround by handfixing it in csv
# - some answers are not processed in unify_answers_iterator (not sure why and fixed manually for now - might not have been rerun after bugfixes?!) (might be fixed!)

# to do:
# - other files [IR]