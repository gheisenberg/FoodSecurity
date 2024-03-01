from concurrent.futures import ProcessPoolExecutor
from functools import  partial
import numpy as np
import pandas as pd
import rasterio
import helper_utils as hu
import math
import time
import pickle
import sys
import os
import preprocessing_config as cfg
import tensorflow as tf
import warnings
import shutil
import geo_utils as gu


def channel_wise_means(file, height, width, channel_l, replace_nan_value, clipping_values, std_multiplier):
    array, nulls, profile = gu.load_geotiff(file, height, width, channel_l, replace_nan_value, clipping_values,
                                            std_multiplier, drop_perc_NaNs=cfg.drop_perc_NaNs,
                                            delete_perc_NaNs=cfg.delete_perc_NaNs)
    if not isinstance(array, bool):
        return np.nanmean(array, axis=(1,2)), nulls, file
    else:
        return False, nulls, file


def channel_wise_std(file, height, width, channel_l, replace_nan_value, clipping_values, std_multiplier, means):
    """returns the means of squares xi - xmean - after adding these take the sq root and you get the standard deviation
    per channel"""
    array, nulls, profile = gu.load_geotiff(file, height, width, channel_l, replace_nan_value, clipping_values,
                                            std_multiplier, drop_perc_NaNs=cfg.drop_perc_NaNs, delete_perc_NaNs=cfg.delete_perc_NaNs)
    if not isinstance(array, bool):
        array = np.array([np.nanmean((array_i - mean_i)**2) for array_i, mean_i in zip(array, means)])
    return array


def calc_mean(filenames, height, width, channel_l, replace_nan_value, clipping_values, std_multiplier):
    """Calculates means of all channels over all files in parallel and then calculates their mean
    --> Working and sufficiently exact (~+-e-07 due to not weighting files by available Numbers <- could be fixed
    but is not necessary imo)"""
    with ProcessPoolExecutor() as executor:
        func = partial(channel_wise_means, height=height, width=width, channel_l=channel_l,
                       replace_nan_value=replace_nan_value, clipping_values=clipping_values,
                       std_multiplier=std_multiplier)
        results = list(executor.map(func, filenames))
    #testing for False does not work with np arrays...
    #ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
    mean_array = np.array([r[0] for r in results if not isinstance(r[0], bool)])
    nulls = {r[2]: [r[1], time.ctime(os.stat(r[2]).st_mtime) if not isinstance(r[0], bool) else False] for r in results}
    means = np.nanmean(mean_array, axis=0)
    print('means', means)
    return means, nulls


def calc_std(filenames, height, width, channel_l, replace_nan_value, clipping_values, std_multiplier, means):
    """Calculates the standard deviation of all channels over all files in parallel and then calculates their standard
    deviation. Needs the means of these channels to be calculated beforehand.
    --> Working and sufficiently exact (~+-0.1% due to not weighting files by available Numbers <- could be fixed
    but is not necessary imo)"""
    with ProcessPoolExecutor() as executor:
        func = partial(channel_wise_std, height=height, width=width, channel_l=channel_l,
                       replace_nan_value=replace_nan_value, clipping_values=clipping_values,
                       std_multiplier=std_multiplier, means=means)
        results = list(executor.map(func, filenames))
    std_array = np.array([r for r in results if not isinstance(r, bool)])
    stds = np.nanmean(std_array, axis=0)
    stds = stds ** (1/2)
    print('stds', stds)
    return stds


def check_man(filenames):
    """Implemented to check the other algorithms. Will crash when loading to many files!!! ~1500 files a 13 channels max
    --> Are working and sufficiently exact (~+-1 due to not weighting files by available Numbers <- could be fixed
    but is not necessary imo"""
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(gu.load_geotiff, filenames, [cfg.height for f in filenames],
                                    [cfg.width for f in filenames],
                                    [cfg.channel_l for f in filenames], [cfg.drop_perc_NaNs for f in filenames],
                                    [cfg.delete_perc_NaNs for f in filenames]))
    results = [r[0] for r in results if not isinstance(r, bool)]
    array = np.array(results)
    print(array.shape)
    means = np.nanmean(array, axis=(0,2,3))
    stds = np.nanstd(array, axis=(0,2,3))
    # ma = np.nanmax(array, axis=(0,2,3))
    print('means2', means)
    print('stds2', stds)
    # print('max', ma)
    # print('maxes', np.nanmax(array, axis=(2,3)))
    return means, stds


def preprocess_file(file, height, width, channel_l, normalization='Z', means=False, stds=False,
                    replace_nan_value='local channel mean', clipping_values=False, std_multiplier=3,
                    write=False, base_p_w=False):
    """Preprocess files by filling values for NaNs, clipping values and normalizing values. Filling needs to be done
    before the rest.

    Args:
        file, height, width, channel_l, replace_nan_value, clipping_values, std_multiplier: from load_geotiff
        normalization: 'Z', 'Z local' or False. 'Z' uses 'Z' normalization and expects channelwise means and stds from
        the whole dataset again, 'Z local' usese local channel wise means and stds and 'Z local all channel'
        normalizes over all channel of one image (not recommended! just testing!)
        means, stds: need to be passed if normalization is 'Z' (and calculated before for the whole dataset)
        write: False - returns array, 'TFR' writes to TFRecord or 'geotiff' writes to geotiff
    """
    if isinstance(file, tf.Tensor):
        file = file.numpy()
        file = bytes.decode(file)
    array, nulls, profile = gu.load_geotiff(file, height, width, channel_l, replace_nan_value, clipping_values,
                                            std_multiplier, drop_perc_NaNs=cfg.drop_perc_NaNs, delete_perc_NaNs=cfg.delete_perc_NaNs)
    #normalizing
    if not isinstance(array, bool):
        if normalization is False:
            pass
        elif normalization == 'Z' or normalization == 'Z local' or normalization == 'Z combined':
            if normalization == 'Z local' or 'Z combined':
                meansl = array.mean(axis=(1,2))
                stdsl = array.std(axis=(1, 2))
                if normalization == 'Z combined':
                    means = (means + meansl)/2
                    stds = (stds + stdsl)/2
                else:
                    means = meansl
                    stds = stdsl
            if means is False or stds is False:
                raise ValueError(f"Missing some values: means {means}, stds {stds}")
            #Actual normalization
            array = np.array([(array_i - mean)/std for array_i, mean, std in zip(array, means, stds)])
        elif normalization == 'Z local all channel':
            mean = array.mean()
            std = array.std()
            #Actual normalization
            array = (array - mean)/std
        else:
            NotImplementedError()
        if not write:
            return array
        elif write == 'geotiff':
            # print(profile, type(profile))
            profile.update(dtype=array.dtype,
                           height=array.shape[1],
                           width=array.shape[2],
                           count=len(cfg.channel_l))
            # print(profile)
            # input('w')
            file_n = os.path.basename(file)
            with rasterio.open(base_p_w + file_n,
                                mode="w",
                                **profile) as new_dataset:
                new_dataset.write(array)
            return None



def main():
    ts = time.time()
    # amount = 100
    for restrict_files in cfg.restrict_files:
        print('Restricting files to', restrict_files)
        print('u = urban, r = rural, all = no restriction')
        preprocessed_p = cfg.preprocessed_p + restrict_files + '/'
        if not os.path.exists(preprocessed_p):
            os.mkdir(preprocessed_p)
        ###Restrict files to labels_df
        if restrict_files == 'U' or restrict_files == 'R':
            labels_df = pd.read_csv(cfg.labels_f)
            print('In',labels_df)
            labels_df = labels_df[labels_df['URBAN_RURA'] == restrict_files]
            labels_df['path'] = cfg.file_p + labels_df['GEID'] + labels_df['DHSID'].str[-8:] + '.tif'
            available_files = hu.files_in_folder(cfg.file_p)
            # check if actually available
            labels_df["path"] = labels_df['path'].apply(lambda x: x if x in available_files else np.NaN)
            labels_df = labels_df[['path', 'DHSYEAR', 'LATNUM', 'LONGNUM',]]

            if labels_df['path'].isna().any():
                print(labels_df[labels_df['path'].isna()])
                warnings.warn(
                    f"!!!Caution Missing values getting dropped {len(labels_df[labels_df['path'].isna()])}"
                    f" from which are {len(labels_df[labels_df['DHSYEAR'] < 2012])} older than 2012"
                    f"--> writing missing files into {preprocessed_p + 'missing_files.csv'}")
                if not labels_df[labels_df['path'].isna()].empty:
                    labels_df[labels_df['path'].isna()].to_csv(preprocessed_p + 'missing_files.csv')
            path_s = labels_df['path'].dropna()
            filenames = list(path_s)
        else:
            filenames = hu.files_in_folder(cfg.file_p)

        ###
        for geotiff_normalization in cfg.geotiff_normalization:
            # filenames = filenames[:amount]
            print('file path', cfg.file_p)
            print('amount of files', len(filenames))
            print('Normalization mode for geotiffs', geotiff_normalization)
            means = False
            stds = False
            outpath = f"{preprocessed_p}{cfg.height}x{cfg.width}_c{''.join(str(c) for c in cfg.channel_l)}" \
                      f"_fill{cfg.fill_method}_r{cfg.replace_nan_value}_clipv{cfg.clipping_values}" \
                      f"{cfg.std_multiplier}_norm{geotiff_normalization}_f{len(filenames)}/"
            print('Outpath', outpath)
            if not os.path.exists(outpath) or cfg.overwrite:
                if os.path.exists(outpath):
                    shutil.rmtree(outpath)
                os.mkdir(outpath)
                if geotiff_normalization == 'Z' or geotiff_normalization == 'Z combined':
                    if cfg.load_means_stds_f and not cfg.overwrite:
                        print('Loading means and stds')
                        with open(cfg.load_means_stds_f, 'rb') as file_pi:
                            (means, stds) = pickle.load(file_pi)
                        print('Loaded means and stds')
                        print(f'channel means {means}, stds {stds}')
                    else:
                        print('Calculating means and stds')
                        means, nulls = calc_mean(filenames, cfg.height, cfg.width, cfg.channel_l, cfg.replace_nan_value,
                                          cfg.clipping_values, cfg.std_multiplier)
                        nulls = dict(sorted(nulls.items(), key=lambda item: item[1][0]))
                        # print('Missing values', nulls)
                        # for k, v in nulls.items():
                        #     print(k, v)
                        with open(os.path.join(preprocessed_p, 'null_values_gtiff'), 'wb') as file_pi:
                            pickle.dump(nulls, file_pi)
                        stds = calc_std(filenames, cfg.height, cfg.width, cfg.channel_l, cfg.replace_nan_value, cfg.clipping_values,
                                        cfg.std_multiplier, means)
                        t2 = time.time()
                        print('Calculated means and stds for Z normalization', (t2 - ts)/60, 'mins')
                        print(f'channel means {means}, stds {stds}')
                        with open(outpath + 'means_stds', 'wb') as file_pi:
                            pickle.dump((means, stds), file_pi)

                ###Write Geotiffs
                t1 = time.time()
                print('starting Normalization')
                warnings.warn("Implementation does not correct the geotransformation for the middle clip "
                              "The geotiff will not have the correct position cf. crop_mid_array for more information")
                func = partial(preprocess_file, height=cfg.height, width=cfg.width, channel_l=cfg.channel_l,
                               normalization=geotiff_normalization,
                               replace_nan_value=cfg.replace_nan_value, clipping_values=cfg.clipping_values,
                               std_multiplier=cfg.std_multiplier, means=means, stds=stds, write='geotiff', base_p_w=outpath)
                with ProcessPoolExecutor() as executor:
                    results = list(executor.map(func, filenames))

                t2 = time.time()
                print('Normalized and wrote geotiffs', (t2 - t1) / 60, 'mins')
            else:
                raise ValueError('Outpath already exists', outpath)
    print('Overall time', (time.time() - ts)/60, 'mins')


if __name__ == "__main__":
    main()
