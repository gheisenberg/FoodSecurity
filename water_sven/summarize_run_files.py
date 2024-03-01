import os
import fnmatch
import pandas as pd

def find_files(path, matching_pattern):
    csv_files = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if fnmatch.fnmatch(file, matching_pattern):
                file_path = os.path.join(root, file)
                csv_files.append(file_path)

    return csv_files


path = '/mnt/datadisk/data/Projects/water/trainH_XV/'
matching_pattern = '*.csv'
csv_files = find_files(path, matching_pattern)
csv_files = [f for f in csv_files if 'run_summary' in f and not 'splits' in f and 'split_' in f]
print(csv_files)
#retrieve the folder name above the folder in which the csv files are located
folders = [os.path.basename(os.path.dirname(os.path.dirname(f))) for f in csv_files]
print(folders)
new_dfs = []
old_dfs = []
all_dfs = []
for split, csv in zip(folders, csv_files):
    df = pd.read_csv(csv)
    df['split name'] = split
    all_dfs.append(df)
    for row, row_df in df.iterrows():
        print(row)
        print(row_df)
        if 'test amount' in row_df and row_df['test amount'] > 100:
            if 'mean(Std test Prediction)' not in row_df:
                new_dfs.append(row_df.to_frame().T)
            else:
                old_dfs.append(row_df.to_frame().T)
old_df = pd.concat(old_dfs)
new_df = pd.concat(new_dfs)
old_df.to_csv(os.path.join(path, 'run_summary_old.csv'), index=False)
new_df.to_csv(os.path.join(path, 'run_summary_new.csv'), index=False)
all_df = pd.concat(all_dfs)
all_df.to_csv(os.path.join(path, 'run_summary_all.csv'), index=False)
