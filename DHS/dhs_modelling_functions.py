import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict
from sklearn.model_selection import KFold
from icecream import ic


def final_ds_droping_cols(df_in, drop_meta=False, drop_6_7_y_FS=True, drop_food_help=True, drop_IPC_wo_food_help=False, drop_perc=75, drop_region=True, 
                drop_languages=['language of interview', 'native language', 'language of questionnaire'],
                IPC_mode='mean', drop_20_25_y_FS=True, drop_0_1_y_FS=True, use_short_long_FS_time_spans='long', numerical_data=['mean', 'std', 'skewness', 'kurtosis'], 
                use_NAN_amount_and_replace_NANs_in_categorical=False, retain_year=True, retain_month=True, 
                retain_adm=['adm0_gaul', 'adm1_gaul', 'adm2_gaul'], retain_GEID_init=True, retain_percentage_of_valid_answers=False,
                drop_single_categorical_cols=True, drop_one_of_double_categorical_cols=True, drop_highly_correlated_cols=False, 
                drop_data_sets=[], drop_agricultural_cols=False, drop_below_version=False, ensure_fold_columns_are_available=True,
                verbose=3):
    """
    This function retrieves a DataFrame after applying a series of transformations based on the provided parameters.

    Args:
        df_in (pd.DataFrame): The input DataFrame.
        drop_meta (bool, optional): Whether to drop metadata columns. Defaults to False.
        drop_6_7_y_FS (bool, optional): Whether to drop food security columns for 6-7 years. Defaults to True.
        drop_food_help (bool, optional): Whether to drop food help columns. Defaults to True.
        drop_IPC_wo_food_help (bool, optional): Whether to drop IPC columns without food help. Defaults to False.
        drop_perc (int, optional): The percentage of NaN values a column can have before it is dropped. Defaults to 45. Not applied on food security columns.
        drop_region (bool, optional): Whether to drop region columns. Defaults to True.
        drop_languages (list, optional): List of languages to drop. Defaults to ['language of interview', 'native language', 'language of questionnaire'].
        IPC_mode (str, optional): The mode of IPC to retain. Defaults to 'mean'.
        drop_20_25_y_FS (bool, optional): Whether to drop food security columns for 20-25 years. Defaults to True.
        drop_0_1_y_FS (bool, optional): Whether to drop food security columns for 0-1 years (overlaps with 0-2 years). Defaults to True.
        use_short_long_FS_time_spans (str, optional): Whether to use short or long food security time spans. Defaults to 'long'.
        numerical_data (list, optional): List of numerical data to retain. Possible: ['mean', 'median', 'std', 'skewness', 'kurtosis']. Defaults to ['mean', 'std', 'skewness', 'kurtosis'].
        use_NAN_amount_and_replace_NANs_in_categorical (bool, optional): Whether to replace NaNs in categorical data with 0. Defaults to True.
        retain_year (bool, optional): Whether to retain year columns. Defaults to True.
        retain_month (bool, optional): Whether to retain month columns. Defaults to True.
        retain_adm (list, optional): List of administrative divisions to retain. Defaults to ['adm0_gaul', 'adm1_gaul', 'adm2_gaul'].
        retain_GEID_init (bool, optional): Whether to retain initial GEID. Defaults to True.
        retain_percentage_of_valid_answers (bool, optional): Whether to retain percentage of valid answers. Defaults to False.
        drop_single_categorical_cols (bool, optional): If True, the function will drop categorical columns with only one category. Default is True. Disregards 'NaN' columns.
        drop_one_of_double_categorical_cols (bool, optional): If True, the function will drop one of the two categorical columns when there are only two categories for a specific feature. The column with fewer non-NaN values will be dropped. Default is True. Disregards 'NaN' columns.
        ensure_fold_columns_are_available (bool, optional): If True, the function will ensure that the columns 'Meta;' + 'adm0_gaul', 'year' and 'GEID_init' are retained. Default is True.
        drop_highly_correlated_cols (bool, optional): If True, the function will drop highly correlated columns. Default is False.
        drop_data_sets (list, optional): List of data sets to drop. Defaults to []. Possible values: ['Meta', 'FS', 'DHS Num', 'DHS Cat', 'Meta one-hot encoding', 'Meta frequency encoding'].
        drop_agricultural_cols (bool, optional): If True, the function will drop agricultural columns, might be suitable for urban calculations. Default is False.
        drop_below_version (int, optional): If set, the function will drop all data below the specified version. Default is False.
        verbose (int, optional): Verbosity level. Defaults to 3.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    
    #               drop_data_sets=['Meta', 'FS', 'DHS Num', 'DHS Cat', 'Meta one-hot encoding', 'Meta frequency encoding'],
    df = df_in.copy()
    df = df.reset_index(drop=False)
    incoming_cols = df.columns.tolist()
    
    # drops all data below versions
    if drop_below_version:
        if not 'version_nr' in df.columns:
            # #create version nr from GEID_init
            df['version_nr'] = df['Meta; GEID_init'].str[4].astype(int).values
        df = df[df['version_nr'] >= drop_below_version]
        df.drop(columns=['version_nr'])
        
    if drop_data_sets:
        drop_data_sets = [c + ';' for c in drop_data_sets]
        # Drop specified datasets:
        for drop_data_set in drop_data_sets:
            cols = [c for c in df.columns if drop_data_set in c]
            df.drop(columns=cols, inplace=True)
            if verbose == 3:
                print(f'Dropped {drop_data_set} data: {cols}')
    
    if drop_meta:
        retaining_inds = []
        if retain_year or ensure_fold_columns_are_available:
            retaining_inds.append('year')
        if retain_month:
            retaining_inds.append('month')
        if retain_GEID_init or ensure_fold_columns_are_available:
            retaining_inds.append('GEID_init')
        if retain_percentage_of_valid_answers:
            retaining_inds.append('percentage of valid answers')
        if retain_adm:
            retaining_inds += retain_adm
        if ensure_fold_columns_are_available and 'adm0_gaul' not in retaining_inds:
            retaining_inds.append('adm0_gaul')
            
        if not retaining_inds:
            cols = [c for c in df.columns if 'Meta; ' in c]# and not 'year' in c]
        else:
            cols = [c for c in df.columns if 'Meta; ' in c and not any(ind in c for ind in retaining_inds)]
        df.drop(columns=cols, inplace=True)
        drop_meta_cols = cols
    if drop_6_7_y_FS:
        cols = [c for c in df.columns if '6-7y' in c and 'IPC' in c]
        df.drop(columns=cols, inplace=True)
    if drop_20_25_y_FS:
        cols = [c for c in df.columns if '20-25y' in c and 'FS; IPC' in c]
        df.drop(columns=cols, inplace=True)
    if drop_0_1_y_FS:
        cols = [c for c in df.columns if '0-1y' in c and 'FS; IPC' in c]
        df.drop(columns=cols, inplace=True)
    if drop_food_help:
        cols = [c for c in df.columns if 'FS; IPC + FH:' in c]
        df.drop(columns=cols, inplace=True)
    if drop_IPC_wo_food_help:
        cols = [c for c in df.columns if 'FS; IPC:' in c]
        df.drop(columns=cols, inplace=True)
    if use_NAN_amount_and_replace_NANs_in_categorical:
        cols = [c for c in df.columns if 'DHS Cat;' in c]
        df[cols] = df[cols].fillna(0.0)
    else:
        cols = [c for c in df.columns if 'DHS Cat;' in c and ': NaN' in c]
        df.drop(columns=cols, inplace=True)
    if drop_perc:
        cols = [c for c in df.columns if len(df[c].dropna()) / len(df[c]) * 100 < drop_perc and not 'FS; IPC' in c]
        df.drop(columns=cols, inplace=True)
        drop_perc_cols = cols
    if drop_region:
        cols = [c for c in df.columns if 'DHS Cat; region:' in c]
        df.drop(columns=cols, inplace=True)
    if drop_languages:
        cols = [c for c in df.columns if any([lang in c for lang in drop_languages])]
        df.drop(columns=cols, inplace=True)
    if IPC_mode:
        cols = [c for c in df.columns if 'FS; IPC' in c and not IPC_mode in c]
        df.drop(columns=cols, inplace=True)
    if use_short_long_FS_time_spans == 'long':
        long_strs = ['0-5m', '6-12m', '1-2y', '2-4y', '4-6y', '6-7y', '6-10y', '10-15y', '15-20y', '20-25y']
        cols = [c for c in df.columns if 'FS; IPC' in c and any([s in c for s in long_strs])]
        df.drop(columns=cols, inplace=True)
    if use_short_long_FS_time_spans == 'short':
        short_strs = ['0-1y', '0-2y', '2-6y', '6-12y', '12-20y']
        cols = [c for c in df.columns if 'FS; IPC' in c and any([s in c for s in short_strs])]
        df.drop(columns=cols, inplace=True)
    if numerical_data:
        cols = [c for c in df.columns if 'DHS Num;' in c and not any([s in c for s in numerical_data])]
        df.drop(columns=cols, inplace=True)
        if verbose == 3:
            print(f'Dropped numerical data: {cols}')
    if drop_agricultural_cols:
        agriculture_cols = ['owns rabbits', 'hectares of agricultural land (1 decimal)', 'owns goats', 'owns horses/ donkeys/ mules','owns cows/ bulls', 'owns chickens/poultry', 'sewing machine',
                   'owns land usable for agriculture', 'owns sheep', 'owns livestock, herds or farm animals']
        drop_cols = []
        for col_a in agriculture_cols:
            for col_df in df.columns:
                if col_a in col_df:
                    drop_cols.append(col_df)
        df.drop(columns=drop_cols, inplace=True)

        
    
    if drop_single_categorical_cols or drop_one_of_double_categorical_cols:
        # Build a dictionary to count the categories
        dic = defaultdict(list)
        for col in df.columns:
            if 'DHS Cat;' in col:
                features = col.rsplit(': ', 1)
                col_base = features[0]
                col_cat = features[1]
                if col_cat != 'NaN':
                    dic[col_base].append(col_cat)
                
        for col_base, l in dic.items():
            if len(l) == 2 and drop_one_of_double_categorical_cols:
                col1 = f'{col_base}: {l[0]}'
                col2 = f'{col_base}: {l[1]}'
                # Drop the category with most NaNs if there are only two - disregarding NaN cols
                if len(df[col1].dropna()) > len(df[col2].dropna()):
                    df = df.drop(columns=[col2])
                    if verbose == 3:
                        print(f'Dropped one of double cat {col2}')
                elif len(df[col1].dropna()) == len(df[col2].dropna()):
                    if 'yes' in l and 'no' in l:
                        col_to_drop = col_base + ': no'
                    elif 'applicable' in l and 'not applicable' in l:
                        col_to_drop = col_base + ': not applicable'
                    else:
                        col_to_drop = col2
                    df = df.drop(columns=[col_to_drop])
                    if verbose == 3:
                        print(f'Dropped one of double cat similar length {col_to_drop}')
                else:
                    df = df.drop(columns=[col1])
                    if verbose == 3:
                        print(f'Dropped one of double cat2 {col1}')
            # Drop the category if there is only one - disregarding NaN cols
            elif len(l) == 1 and drop_single_categorical_cols:
                df = df.drop(columns=[f'{col_base}: {l[0]}'])
                if verbose == 3:
                    print(f'Dropped single cat {col_base}: {l[0]}')
    
    if drop_highly_correlated_cols:
        df = drop_highly_correlated_cols_f(df, drop_highly_correlated_cols, verbose)
        
    if verbose:
        num_dropped = set(['mean', 'median', 'std', 'skewness', 'kurtosis']).difference(numerical_data)
        print(f'Dropped {len(incoming_cols) - len(df.columns)}, retaining columns: {len(df.columns)}')
        print(f'Dropped following subsets of numerical data: {num_dropped}')
        print(f'Retained following subsets of food security data: \n   {IPC_mode} \n   {use_short_long_FS_time_spans} time spans.\nDropped 6-7y {drop_6_7_y_FS} \n    20-25y {drop_20_25_y_FS}\n'
              f'Dropped: food help {drop_food_help}\n    IPC w/o food help {drop_IPC_wo_food_help}')
        print(f'Dropped meta data {drop_meta} and region {drop_region}')
        print(f'Dropped columns with less than {drop_perc}% of available values')
        
        if verbose > 1:
            seen = set()
            for c in drop_meta_cols:
                c_base = c
                if ': ' in c_base:
                    c_base = c.rsplit(': ', 1)[0]
                if c_base not in seen:
                    try:
                        print(f'Dropped meta: {c}: {round(len(df_in[c].dropna()) / len(df_in[c]) * 100, 1)}%')
                        seen.add(c_base)
                    except:
                        print('An index column was dropped, probably GEID_init or adm2_gaul or cluster number')
            for c in drop_perc_cols:
                c_base = c
                if ': ' in c_base:
                    c_base = c.rsplit(': ', 1)[0]
                if c_base not in seen:
                    try:
                        print(f'Dropped meta: {c}: {round(len(df_in[c].dropna()) / len(df_in[c]) * 100, 1)}%')
                        seen.add(c_base)
                    except:
                        print('An index column was dropped, probably GEID_init or adm2_gaul or cluster number')
            
            print('Remaining columns:')
            seen = set()
            for c in df.columns:
                c_base = c
                if ': ' in c_base:
                    c_base = c.rsplit(': ', 1)[0]
                if c_base not in seen:
                    if verbose == 2:
                        print(f'Remaining: {c}: {round(len(df[c].dropna()) / len(df[c]) * 100, 1)}% (and subsets)')
                        seen.add(c_base)
                    else:
                        print(f'Remaining: {c}: {round(len(df[c].dropna()) / len(df[c]) * 100, 1)}%')

    return df


def drop_highly_correlated_cols_f(df, drop_highly_correlated_cols, verbose):
    # Select only numerical columns
    df_numeric = df.select_dtypes(include=[np.number])
    
    # throw away further cols
    df_numeric = df_numeric.drop(columns=[c for c in df_numeric.columns if c[:4] == 'Meta'])
    df_numeric = df_numeric.drop(columns=[c for c in df_numeric.columns if 'DHS Num;' in c and c[-6:] == 'median'])

    # Calculate the correlation matrix
    corr_matrix = df_numeric.corr().abs()

    # Create a mask for the upper triangle
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)

    # Apply the mask to the correlation matrix
    upper = corr_matrix.where(mask)

    # Filter the correlation matrix to only include values greater than 0.5
    filtered_corr_matrix = upper.stack().where(lambda x: x > 0.5).dropna()

    # Get the columns to keep (those that have at least one correlation above 0.5)
    cols_to_keep = filtered_corr_matrix.index.get_level_values(0).union(filtered_corr_matrix.index.get_level_values(1)).unique()

    # Drop the columns from df_numeric that are not in cols_to_keep
    df_numeric = df_numeric[cols_to_keep]

    if verbose >= 4:
        # Convert the Series back to a DataFrame and unstack it
        filtered_corr_matrix = pd.DataFrame(filtered_corr_matrix, columns=['correlation']).unstack()
        
        # Create a heatmap of the filtered correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(filtered_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Heatmap')
        plt.show()

    to_drop = []
    for column in upper.columns:
        high_corr = [col for col in upper.columns if upper[col][column] > drop_highly_correlated_cols]
        if high_corr:
            # Compare the count of non-NaN values in the columns
            non_nan_counts = df_numeric[[column] + high_corr].count()
            column_to_keep = non_nan_counts.idxmax()  # Get the column with the most non-NaN values
            # Add all other columns to the to_drop list
            to_drop.extend([col for col in [column] + high_corr if col != column_to_keep])

    df = df.drop(columns=to_drop)
    if verbose >= 3:
        for col in to_drop:
            print(f'Dropped highly correlated columns: {col}')
            
    return df


def fold_generator(data, split_type, n_splits=5, verbose=1):
    """
    Generate indices for train and test sets based on the specified split type.

    Parameters:
    data (DataFrame): The input dataset.
    split_type (str): The type of split - 'country', 'survey', or 'year'.
    n_splits (int): Number of splits/folds for the outer cross-validation.
    verbose (int): Level of verbosity.
    """
    if split_type == 'country':
        split_col = 'Meta; adm0_gaul'
    elif split_type == 'survey':
        split_col = 'Meta; GEID_init'
    elif split_type == 'year':
        split_col = 'Meta; rounded year'
        # Ensure 'Meta; rounded year' column is created outside this function or create here based on logic provided
        data[split_col] = data.groupby('Meta; GEID_init')['Meta; year'].transform(lambda x: round(x.mean()))
    elif split_type == 'unconditional':
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(data):
            yield data.index[train_idx], data.index[test_idx]
        return
    else:
        raise ValueError(f'Invalid split_type: {split_type}')

    unique_combinations = data[split_col].drop_duplicates().values
    
    # Adjust maximum n_splits based on the number of unique combinations
    if len(unique_combinations) < n_splits or n_splits == -1:
        n_splits = len(unique_combinations)
        if verbose:
            ic(f'Adjusting n_splits to the length of unique combinations ({n_splits}) for', split_type)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_idx, test_idx in kf.split(unique_combinations):
        train_combinations = unique_combinations[train_idx]
        test_combinations = unique_combinations[test_idx]
        
        train_mask = data[split_col].isin(train_combinations)
        test_mask = data[split_col].isin(test_combinations)

        # Yielding the indices for train and test sets
        yield data[train_mask].index, data[test_mask].index
