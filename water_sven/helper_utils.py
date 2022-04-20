import os
from itertools import chain, combinations
import pickle

#own imports
import nn_utils as nnu


def paths_from_base_path(base_path, folders_d, verbose=1):
    """Returns subfolders from a base folder (creates them if necessary)

    Args:
        base_path (str): File path from where to start creating paths from. Usually the base_folder defined in config.py
        folders_d (dict): Dictionary of additional folder names:
            {internal name in code: [their relative path to be created from, folder name, indicator for generating
                unique path]}
            e.g. {'sentinel_path': ('base_path', 'Sentinel2/', False)
        verbose (int): Use 0 to silence prints and 1 or higher to get more verbosity

    Returns:
        paths_l (list): A list of paths (str) created
    """
    paths_l = []
    for name, [rel_folder_name, rel_path, indicator] in folders_d.items():
        ###Folders
        path_b = base_path if rel_folder_name == 'base_path' else folders_d[rel_folder_name][-1]
        if type(path_b) != str:
            raise TypeError("path is", path_b, "Should be a path though")
        path = os.path.join(path_b, rel_path)
        if indicator:
            #add stuff to path so it wont exist
            nr = 1
            while os.path.exists(path + '_' + str(nr)):
                nr += 1
            path += '_' + str(nr) + '/'
        if not os.path.exists(path):
            os.mkdir(path)
        paths_l.append(path)
        #append the result to dict to reload it when necessary
        folders_d[name].append(path)
    if verbose >= 1:
        print('Loaded paths', paths_l)
    return paths_l


def load_class_weights(class_weights_f, labels_df, train_path, prj_path, load_data, verbose=1):
    """Loads class_weights from file or calls the routine to calculate them

    Args:
        class_weights_f (str): File path to load class_weights from or save them
        labels_df (pandas DF): Pandas dataframe with label information
        train_path (str): Folder path where training images are located
        prj_path (str): Folder path where the project files are located
        load_data (bool): True or False, use False if you want to recalculate the class_weights
        verbose (int): Use 0 to silence prints and 1 or higher to get more verbosity

    Returns:
        class_weights (dict): Weights for every class
    """
    if os.path.exists(class_weights_f) and load_data:
        class_weights = pickle.load(open(class_weights_f, "rb"))
        if verbose >= 1:
            print('Loaded class weights from file', class_weights)
    else:
        # Create class weights and save them to file
        class_weights = nnu.create_class_weights_dict(train_path, labels_df)
        if verbose >= 1:
            print('Created class weights', class_weights)
        # Save class_weights as pickle
        with open(os.path.join(prj_path, 'class_weights'), 'wb') as file_pi:
            pickle.dump(class_weights, file_pi)
    return class_weights


def powerset(iterable):
    """Creates a powerset (cf. returns)

    Args:
        iterable (iterable): List, dict, set...

    Returns:
        powerset (iterable): ([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
