import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict
from sklearn.model_selection import KFold
from icecream import ic
from sklearn.metrics import r2_score, mean_squared_error
from pdpbox import pdp, get_example, info_plots
import plotly.io as py
from sklearn.model_selection import train_test_split



def final_ds_droping_cols(df_in, drop_meta=False, drop_6_7_y_FS=True, drop_food_help=True, drop_IPC_wo_food_help=False, drop_perc=75, drop_region=True, 
                drop_languages=['language of interview', 'native language', 'language of questionnaire'],
                IPC_mode='mean', drop_20_25_y_FS=True, drop_0_1_y_FS=True, use_short_long_FS_time_spans='long', numerical_data=['mean', 'std', 'skewness', 'kurtosis'], 
                use_NAN_amount_and_replace_NANs_in_categorical=False, retain_year=True, retain_month=True, 
                retain_adm=['adm0_gaul', 'adm1_gaul', 'adm2_gaul'], retain_GEID_init=False, retain_percentage_of_valid_answers=False,
                drop_single_categorical_cols=True, drop_one_of_double_categorical_cols=True, drop_highly_correlated_cols=False, 
                drop_data_sets=[], drop_agricultural_cols=False, drop_below_version=False, ensure_fold_columns_are_available=True,
                numerical_return_format='float32', retrieve_ds=[],
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
        only_ds
        drop_agricultural_cols (bool, optional): If True, the function will drop agricultural columns, might be suitable for urban calculations. Default is False.
        drop_below_version (int, optional): If set, the function will drop all data below the specified version. Default is False.
        verbose (int, optional): Verbosity level. Defaults to 3.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    
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
    
    if retrieve_ds:
        all_ds = set([c.split(';', 1)[0] for c in df.columns])
        print(all_ds)
        print(set(retrieve_ds))
        drop_data_sets = all_ds.difference(set(retrieve_ds))
        print(drop_data_sets)
        if len(drop_data_sets) == len(all_ds):
            raise ValueError(f'No data sets left after filtering - not matching only_ds: {retrieve_ds}')
           
    if drop_meta or 'Meta' in drop_data_sets:
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
        remaining_meta_cols = [c for c in df.columns if 'Meta; ' in c]
        
    
    if drop_data_sets:
        drop_data_sets = [c + ';' for c in drop_data_sets]
        # Drop specified datasets:
        for drop_data_set in drop_data_sets:
            cols = [c for c in df.columns if drop_data_set in c and c not in remaining_meta_cols]
            df.drop(columns=cols, inplace=True)
            if verbose == 3:
                print(f'Dropped {drop_data_set} data: {cols}')
    
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
    
    if 'index' in df.columns:
        df = df.drop(columns=['index'])
    
    if drop_highly_correlated_cols:
        df = drop_highly_correlated_cols_f(df, drop_highly_correlated_cols, verbose)
        
    if numerical_return_format:
        num_cols = df.select_dtypes(include=[np.number])
        df[num_cols.columns] = num_cols.astype(numerical_return_format)
        
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


def fold_generator_3_indices(data, split_type, n_splits=5, verbose=1, val_size=0.2):
    """
    Generate indices for train, validation and test sets based on the specified split type.

    Parameters:
    data (DataFrame): The input dataset.
    split_type (str): The type of split - 'country', 'survey', or 'year'.
    n_splits (int): Number of splits/folds for the outer cross-validation.
    verbose (int): Level of verbosity.
    test_size (float): Proportion of the dataset to include in the test split.
    val_size (float): Proportion of the dataset to include in the validation split.
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
        for train_val_idx, test_idx in kf.split(data):
            # Split the train_val indices into training and validation indices
            train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size, random_state=42)
            yield data.index[train_idx], data.index[val_idx], data.index[test_idx]
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
    
    for train_val_combinations, test_combinations in kf.split(unique_combinations):
        train_val_mask = data[split_col].isin(unique_combinations[train_val_combinations])
        test_mask = data[split_col].isin(unique_combinations[test_combinations])
        train_val_indices = data[train_val_mask].index.values
        test_indices = data[test_mask].index.values
        # Split the train_val indices into training and validation indices
        train_indices, val_indices = train_test_split(train_val_indices, test_size=val_size, random_state=42)
        
        # Yielding the indices for train, validation and test sets
        yield train_indices, val_indices, test_indices
        

def metrices_weighted_available_data(results_df, missing_mask, drop_perc=0, verbose=1):
    """
    Calculate the percentage of available data for each column in the dataset.

    Parameters:
    df (DataFrame): The input dataset.
    verbose (int): Level of verbosity.
    """
    # Calculate the percentage of available data for each row
    available_data = (len(missing_mask.columns) - missing_mask.sum(axis=1)) / len(missing_mask.columns) * 100
    # Assuming df is your DataFrame
    available_data = available_data[results_df.index]

    # Calculate weighted averages of RMSE, nRMSE, R2 and Correlation
    results_df['Available data'] = available_data
    
    res_d = defaultdict(dict)
    #Calculate means, std, weighted means and weighted std for grouped objects an by available data
    for i in range(90, drop_perc - 10, -10):
        
        results_sub_df = results_df[(results_df["Available data"] >= i)]
        print('asd', i, len(results_sub_df))
        mse = mean_squared_error(results_sub_df['Actual'], results_sub_df['Prediction'])
        rmse = np.sqrt(mse)
        r2 = r2_score(results_sub_df['Actual'], results_sub_df['Prediction'])
        corr = np.corrcoef(results_sub_df['Actual'], results_sub_df['Prediction'])[0, 1]
        nrmse = rmse / np.std(results_sub_df['Actual'])
        
        if len(results_df) == len(results_sub_df):
            i = 'Overall'
        res_d[i] = [rmse, nrmse, r2, corr]
        if i == 'Overall':
            break
    
    #Create DataFrame from dictionary
    res_df = pd.DataFrame.from_dict(res_d, orient='index', columns=['RMSE', 'nRMSE', 'R2', 'Correlation']).reset_index()
    res_df = res_df.reset_index()
    return res_df

def create_history_figures(history, out_f, fold, write_out_f=False):
    print(history.history.keys())
    print(history.history['loss'])
    plt.figure(figsize=(10, 6), facecolor='white')
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve', fontsize=20)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(fontsize=12)
    if write_out_f:
        plt.savefig(f'{out_f}loss_curve_fold{fold+1}.png')
    plt.show()
        
    plt.figure(figsize=(10, 6), facecolor='white')
    plt.plot(history.history['root_mean_squared_error'], label='Training RMSE')
    plt.plot(history.history['val_root_mean_squared_error'], label='Validation RMSE')
    plt.title('RMSE Curve', fontsize=20)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('RMSE', fontsize=16)
    plt.legend(fontsize=15)
    if write_out_f:
        plt.savefig(f'{out_f}rmse_curve_fold{fold+1}.png')
    plt.show()
    

def create_scatterplot(df, out_f):    
    mse = mean_squared_error(df['Actual'], df['Prediction'])
    rmse = np.sqrt(mse)
    r2 = r2_score(df['Actual'], df['Prediction'])
    corr = np.corrcoef(df['Actual'], df['Prediction'])[0, 1]
    nrmse = rmse / np.std(df['Actual'])
    plt.figure(figsize=(10, 6), facecolor='white')
    plt.scatter(df['Actual'], df['Prediction'], alpha=0.5)
    plt.xlabel('Actual', fontsize=16)
    plt.ylabel('Prediction', fontsize=16)
    plt.title(f'Prediction vs. Actual', fontsize=20)
    plt.plot([min(df['Actual']), max(df['Actual'])], [min(df['Actual']), max(df['Actual'])], color='red')
    plt.text(min(df['Actual']), max(df['Actual']), f'R²: {r2:.2f}\nRMSE: {rmse:.2f}\nnRMSE: {nrmse:.2f}\nCorr: {corr:.2f}', verticalalignment='top', horizontalalignment='left', backgroundcolor='white', fontsize=15)
    plt.savefig(f"{out_f}")


def create_PDP_plots(X_test, model, out_dir, fold):
    print(X_test.nunique())
    for col in X_test.columns:
        X_in = X_test.copy()
        print(col)
        print('Shape', X_in.shape)
        print(X_in[col].nunique())
        if X_in[col].nunique() <= 1:
            print(X_in[col])
            continue
        if 'encoding' in col:
            continue
        # Create the pdp data to be plotted
        pdp_dist = pdp.PDPIsolate(model=model, df=X_in, model_features=X_in.columns, feature=col, feature_name=col, n_classes=0, num_grid_points=10)
        # plot the PDP for feature 'Distance Covered (Kms)'
        fig, axes = pdp_dist.plot(
            center=False,
            plot_lines=True,
            frac_to_plot=100,
            cluster=False,
            n_cluster_centers=None,
            cluster_method='accurate',
            plot_pts_dist=True,
            to_bins=False,
            show_percentile=False,
            which_classes=None,
            figsize=None,
            dpi=300,
            ncols=2,
            plot_params={"pdp_hl": True},
            engine='plotly',
            template='plotly_white')

        #save figures
        col_n = col.replace(';', '_').replace(' ', '_').replace('/', '_')
        py.write_image(fig, f'{out_dir}PDP_{col_n}_Fold{fold}.png')
        # fig.show()
