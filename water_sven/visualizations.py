import os
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import pickle

import helper_utils as hu
# import water_w_regression as wwr
import config as cfg
logger = hu.setup_logger(cfg.logging)

def annot_max(x, y, type_m, ax=None):
    """
    Annotate maximum of graph and last point of graph

    Args:
        x (np array): X values of graph
        y (np array): Y values of graph

    Returns:
        None
    """
    if type_m == 'categorical':
        annotations = [(x[np.argmax(y)], max(y)), (x[-1], y[-1])]
    else:
        annotations = [(x[np.argmin(y)], min(y)), (x[-1], y[-1])]
    for (xmax, ymax) in annotations:
        text = "{:.2f}".format(ymax)
        if not ax:
            ax=plt.gca()
        ax.annotate(text, xy=(xmax, ymax))


def plot_history(history, save_path, type_m, verbose=1):
    """
    Visualize the history object

    Args:
        history (Keras history): History object
        save_path (str): Path where to save visualizations
        verbose (int): Use 0 to silence prints and 1 or higher to get more verbosity

    Returns:
        None
    """
    if verbose:
        logger.debug('history %s', history)
        logger.debug(history.history.items())
    for key, metric in history.history.items():
        key_n = key.replace('_', ' ').title()
        if key[:3] != 'val':
            plt.figure()
            plt.grid(which='both', axis='y')
            plt.plot(history.history[key])
            if key != 'lr':
                plt.plot(history.history['val_' + key])
                annot_max(np.arange(len(history.history['val_' + key])), history.history['val_' + key], type_m)
            plt.title('Model ' + key_n)
            plt.ylabel(key_n)
            plt.xlabel('Epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            if key == 'loss':
                plt.ylim(0, 2)
            elif key == 'accuracy':
                plt.ylim(0, 1)
            #plt.show()
            #Save
            plt.tight_layout()
            plt.savefig(save_path + key)
            plt.close()
            #asd


def standard_hist_from_df(df, path, file_n, title_in, bins_count='auto',
                          minv=False, maxv=False, title_add='stats', xlabel='PC1', ylabel='count',
                          xlim=3.5, xticks=1):
    if minv or maxv:
        df = df[(df > minv) & (df < maxv)]
    if bins_count == 'auto':
        bins_count = min(max(int(len(df) / 10), 7), 50)
    df.hist(bins=bins_count)#, density=1)
    if xlim:
        plt.xlim(-xlim, xlim)
    # if xticks:
    #     plt.xticks(np.arange(int(df.min()), int(df.max()), step=xticks))
    # write len, mean std as header to plot
    title = str(title_in)
    if title_add == 'stats':
        title += f"\nn={len(df)}, mean={df.mean():.2f}, std={df.std():.2f}, skew={df.skew():.2f}, kurt={df.kurtosis():.2f}"
    plt.title(title)#, pad=20)
    #label x axis
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path + ' ' + file_n + ' ' + str(title_in) + '.png')
    plt.close()


def plot_DBSCAN(df_clustered, path, file_n):
    # Plot the results
    plt.figure(figsize=(10, 10))
    plt.scatter(df_clustered['LONGNUM'], df_clustered['LATNUM'], c=df_clustered['clustered'], cmap='viridis')
    plt.title('DBSCAN Clustering')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.savefig(path + file_n + '.png')
    plt.close()


from itertools import cycle


def plot_dataframe(df, x_column, y_columns, run_path, file_name, title, label_reduce=False,
                   color_keywords=None, linestyle_keywords=None, xlabel='km', ylabel=False):
    """
    Plot selected columns from a DataFrame.

    Parameters:
    df (DataFrame): DataFrame containing the data to plot.
    x_column (str): Column name to use for x-axis.
    y_columns (list): List of column names to plot on y-axis.
    """

    # Create a new figure
    plt.figure()
    if len(y_columns) > 7:
        plt.figure(figsize=(15, 10))

    # Define color palette
    color_palette = cm.get_cmap('tab10', len(y_columns))

    # Create a cycle of colors
    color_cycle = cycle(range(len(y_columns)))

    # Create a dictionary to hold assigned colors for labels
    label_color_dict = {}

    # Create dictionaries for color and linestyle keywords if provided
    color_keywords_dict = {keyword: next(color_cycle) for keyword in color_keywords} if color_keywords else {}
    linestyle_list = ['-', '--', ':', '-.']
    linestyle_keywords_dict = {keyword: linestyle_list[i % len(linestyle_list)] for i, keyword in
                               enumerate(linestyle_keywords)} if linestyle_keywords else {}
    # Loop over the columns and add each one to the plot
    for y_column in y_columns:
        label = y_column
        if label_reduce:
            label = label.replace(label_reduce, "")

        # Assign color based on label
        color_index = next((color_keywords_dict[keyword] for keyword in color_keywords_dict if keyword in label),
                           label_color_dict.get(label, next(color_cycle)))
        label_color_dict[label] = color_index

        # Assign linestyle based on label
        linestyle = next((linestyle_keywords_dict[keyword] for keyword in linestyle_keywords_dict if keyword in label),
                         '-')

        plt.plot(df[x_column], df[y_column], label=label, color=color_palette(color_index), linestyle=linestyle)

    # Add a legend to the plot
    plt.legend()
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.title(title)

    # Save the plot
    plt.savefig(run_path + file_name + ' ' + title + '.png')
    plt.close()


def plot_dataframe_old(df, x_column, y_columns, run_path, file_name, title, label_split_at_space=False, dashed_lines=False):
    """
    Plot selected columns from a DataFrame.

    Parameters:
    df (DataFrame): DataFrame containing the data to plot.
    x_column (str): Column name to use for x-axis.
    y_columns (list): List of column names to plot on y-axis.
    """

    # Create a new figure
    plt.figure()

    # Loop over the columns and add each one to the plot
    for i, y_column in enumerate(y_columns):
        label = y_column
        if label_split_at_space:
            label = label.rsplit(' ', label_split_at_space)[0]
        linestyle = '--' if dashed_lines and dashed_lines[i] else '-'
        plt.plot(df[x_column], df[y_column], label=label, linestyle=linestyle)

    # Add a legend to the plot
    plt.legend()
    plt.title(title)

    # Show the plot
    plt.savefig(run_path + file_name + title + '.png')
    plt.close()


# def histogram(df, stat, title, path, file_n, limit_above=120):
#     plt.figure()
#     if not type(max(df)) == str and max(df) >= limit_above:
#         bins = list(range(-4, 101, 5))
#         ax = sns.displot(df, stat=stat, bins=bins)
#     else:
#         ax = sns.displot(df, stat=stat)
#     if type(max(df)) == str:
#         ax.set_xticklabels(rotation=90)
#         plt.xlabel('')
#     plt.title(title)
#     # plt.show(bbox_inches='tight')
#     plt.savefig(path + file_n, bbox_inches='tight')
#     plt.close()

#
# # plot as scatter plot and compute a regression line, including alpha and beta
# def scatterplotRegression(df, run_path, file_name=''):
#     """Plots a Regressionplot with a text annotation containing additional information
#     Args:
#         df (pd.DataFrame): DataFrame consisting of the Actual and the Predicted Volatility Values
#         model_type (str): Name of the model
#         EPOCHS (int): the amount of epochs, the model was trained on
#         BATCH_SIZE (int): number of training examples
#         It does not return anything, but plots a regression plot and saves it
#     """
#     beta, alpha = np.polyfit(df.iloc[:,0], df.iloc[:,1], 1)
#     corr_value = df.corr(method="pearson")
#     pearson_corr = corr_value.iloc[1,0]
#     rmse = ((df.iloc[:, 0] - df.iloc[:, 1]) ** 2).mean() ** .5
#     parameter_text = "\n".join(
#         (
#             r"$pearson\ corr=%.2f$" % (pearson_corr,),
#             r"$slope: \beta=%.2f$" % (beta,),
#             r"$intercept: \alpha=%.2f$" % (alpha,),
#             r"$rmse: \alpha=%.2f$" % (rmse,),
#         )
#     )
#     logger.debug(file_name)
#     logger.debug(parameter_text)
#     plt.figure()
#     sns.set(rc={"figure.figsize": (15, 10)})
#     sns.regplot(
#         data=None,
#         x=df.iloc[:,0],
#         y=df.iloc[:,1],
#         x_ci="sd",
#         scatter=True,
#         line_kws={"color": "red"},
#     )
#     # add text annotation
#     # set the textbox color to purple
#     purple_rgb = (255.0 / 255.0, 2.0 / 255.0, 255.0 / 255.0)
#     plt.grid(True)
#     plt.annotate(
#         parameter_text,
#         xy=(0.03, 0.8),
#         xycoords="axes fraction",
#         ha="left",
#         fontsize=14,
#         # color=purple_rgb,
#         backgroundcolor="w",
#     )
#     plt.title('Scatterplot', fontsize=14)
#     plt.xlabel('Actual', fontsize=14)
#     plt.ylabel('Prediction', fontsize=14)
#     # finally save the chart to disk
#     plt.savefig(run_path + 'Scatterplot_' + file_name + ".png", dpi=300, bbox_inches="tight")
#     # plt.show()
#     plt.close()
#     return beta, alpha, pearson_corr, rmse


def scatterplotRegressionMultiInputs(df, run_path, file_name='', multidataset_col=False,
                                     error_metrics_on_test_data=False):
    """Plots a Regressionplot with a text annotation containing additional information
    Args:
        df (pd.DataFrame): DataFrame consisting of the Actual and the Predicted Volatility Values
        model_type (str): Name of the model
        EPOCHS (int): the amount of epochs, the model was trained on
        BATCH_SIZE (int): number of training examples
        It does not return anything, but plots a regression plot and saves it
    """
    df_l = []
    # colors = ['r', 'b', 'g', 'y', 'm', 'c']  # List of colors for each predicted column
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    unq_l = []
    if multidataset_col:
        for unq in df[multidataset_col].unique():
            df_l.append(df[df[multidataset_col] == unq])
            unq_l.append(unq)
    else:
        df_l.append(df)
        unq_l.append('all')

    metrics_df = df
    if error_metrics_on_test_data:
        metrics_df == df[df[multidataset_col] == 'test']

    beta, alpha = np.polyfit(metrics_df.iloc[:,0], metrics_df.iloc[:,1], 1)
    corr_value = metrics_df.iloc[:,:2].corr(method="pearson")
    pearson_corr = corr_value.iloc[1,0]
    rmse = ((metrics_df.iloc[:, 0] - metrics_df.iloc[:, 1]) ** 2).mean() ** 0.5
    nrmse = rmse / metrics_df.iloc[:, 0].std()

    parameter_text = "\n".join(
        (
            r"$pearson\ corr=%.2f$" % (pearson_corr,),
            r"$\alpha=%.2f,\ beta=%.2f$" % (alpha, beta,),
            r"$(n)rmse=%.2f\:\ (%.2f)$" % (rmse, nrmse),
        )
    )
    logger.debug(file_name)
    logger.debug(parameter_text)
    if not multidataset_col:
        sns.set(rc={"figure.figsize": (5, 4)})
        plt.figure()
        plt.annotate(
            parameter_text,
            xy=(0.03, 0.8),
            xycoords="axes fraction",
            ha="left",
            fontsize=10,
            # color=purple_rgb,
            backgroundcolor="w",
        )
    else:
        sns.set(rc={"figure.figsize": (15, 10)})
        plt.figure()
        plt.annotate(
            parameter_text,
            xy=(0.03, 0.8),
            xycoords="axes fraction",
            ha="left",
            fontsize=14,
            # color=purple_rgb,
            backgroundcolor="w",
        )

    plt.grid(True)

    for i, df in enumerate(df_l):
        sns.regplot(
            data=df,
            x=df.columns[0],
            y=df.columns[1],
            x_ci="sd",
            scatter=True,
            color=colors[i],
            label=unq_l[i],
            # line_kws={"color": colors[i]},  # Use a different color for each predicted column
        )

    plt.title(f'Scatterplot (n={len(df)})', fontsize=14)
    plt.xlabel('Actual', fontsize=14)
    plt.ylabel('Prediction', fontsize=14)
    if multidataset_col:
        plt.legend(loc='lower right')
    # finally save the chart to disk
    plt.savefig(run_path + 'Scatterplot: ' + file_name + ".png", dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close('all')
    return beta, alpha, pearson_corr, rmse, nrmse


def plot_CM(label_true, prediction_true, classes, save_path, mode='both', title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Prints and plots the confusion matrix.

    Args:
        label_true (np array): The actual label/ground truth
        prediction_true (np array): The prediction
        classes (list): Label names (ordered by quantity)
        save_path (str): The path where to save the image
        mode (str): If you want a normalized or absolute or both CM ('normalized', 'quantities' or 'both')
        title (str): Title of the image
        cmap (plt.cm.Blues): Colors

    Returns:
        None
    """
    if mode == 'both':
        cm_l = ['quantities', 'normalized']
    elif mode == 'normalized':
        cm_l = ['normalized']
    else:
        cm_l = ['quantities']
    for i in cm_l:
        logger.debug('Creating CM', i)
        plt.figure()
        # plt.figure(figsize=(7, 4))
        cm = confusion_matrix(y_true=label_true, y_pred=prediction_true)
        tick_marks = np.arange(len(classes))
        if len(tick_marks) <= 3:
            plt.xticks(tick_marks, classes, wrap=True)
            plt.yticks(tick_marks, classes, rotation='vertical', wrap=True, verticalalignment='bottom',
                       horizontalalignment='center')
            plt.ylabel('True label', {'size': 'large'})
            plt.xlabel('Predicted label', {'size': 'large'})
        else:
            plt.xticks(tick_marks, classes, rotation='vertical', wrap=True, horizontalalignment='right',
                       verticalalignment='top')
                       # horizontalalignment='center')
            plt.yticks(tick_marks, classes, rotation='horizontal', wrap=True)\
                # , wrap=True, verticalalignment='bottom',
                #        horizontalalignment='center')

        thresh = 0.6
        if i == 'normalized':
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.around(cm, decimals=2)
            logger.debug("Normalized confusion matrix")
        else:
            thresh = thresh * len(label_true)
            logger.debug('Confusion matrix, without normalization')

        logger.debug('threshold for CM', thresh)
        for k, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, k, cm[k, j],
                horizontalalignment="center", size='large',
                color="white" if cm[k, j] > thresh else "black")
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(save_path + i)
        plt.show()
        plt.close()


def plot_CM_sns(label_true, prediction_true, classes, save_path, mode='normalized', title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Prints and plots the confusion matrix.

    Args:
        label_true (np array): The actual label/ground truth
        prediction_true (np array): The prediction
        classes (list): Label names (ordered by quantity)
        save_path (str): The path where to save the image
        mode (str): If you want a normalized or absolute or both CM ('normalized', 'quantities' or 'both')
        title (str): Title of the image
        cmap (plt.cm.Blues): Colors

    Returns:
        None
    """
    if mode == 'both':
        cm_l = ['quantities', 'normalized']
    elif mode == 'normalized':
        cm_l = ['normalized']
    else:
        cm_l = ['quantities']
    for i in cm_l:
        logger.debug('Creating CM', i)
        plt.figure()
        cm = confusion_matrix(y_true=label_true, y_pred=prediction_true)

        if i == 'normalized':
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.around(cm, decimals=2)
            logger.debug("Normalized confusion matrix")
        else:
            logger.debug('Confusion matrix, without normalization')

        # df_cm = pd.DataFrame(cm, index=classes,
        #                      columns=classes)
        # plt.figure(figsize=(10, 7))
        # sns.heatmap(cm, annot=True)
        #from nele
        heatmap = sns.heatmap(cm, vmin=0, vmax=1, annot=True, cmap="PiYG")
        # sns.set(font_scale=1)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, wrap=True)
        plt.yticks(tick_marks, classes, rotation='vertical', wrap=True, verticalalignment='bottom',
                   horizontalalignment='left')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        plt.savefig(save_path + '_sns_' + i)
        plt.close()


def plot_images(images_arr, save_path=False, show=False, title=False):
    """
    Plots and saves images (usually the actual input image into the model)

    Args:
        images_arr (np array): Input image array
        save_path (str): The path where to save the image (None if it shall not be saved)
        show (bool): True if you want python to actually print the image
        title (str): Title of the image

    Returns:
        None
    """
    for nr, img in enumerate(images_arr):
        img = zero_one_normalization(img.copy())
        fig, ax = plt.subplots()
        #ax.imshow(img, interpolation='none')
        ax.imshow(img)
        plt.title(title)
        if save_path:
            #not sure if necessary
            # img = np.rint(img * 256)
            #savefig before show, otherwise img gets empty
            plt.savefig(save_path + str(nr))
        if show:
            plt.show()


def zero_one_normalization(data):
    """
    Returns 0,1 normalized data of arrays - used to normalize images for visualization

    Args:
        data (np array): Input data

    Returns:
        data (np array): Normalized data"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def main():
    run_path = '/mnt/datadisk/data/Projects/water/trainH_XV/split__out_of_country_all_w_TIF/PCA_w_location_weighting_all2/996x996_c432_fillmean_m2.5_rlocal_channel_mean_clipvoutlier_normZ_f31213vgg19_wimagenet_unfl100_d0_lr0.0001_momentum0.9_optimizerSGD__m_vgg19_2/'

    names = []
    val_dfs = []
    test_dfs = []

    for file in hu.files_in_folder(run_path, sort=True):
        print(file)
        if 'val_df_' in file:
            val_dfs.append(pd.read_csv(file))
            names.append(file[-11:-4])
        elif 'test_df_' in file:
            test_dfs.append(pd.read_csv(file))
    print(names)
    print('val\n', val_dfs)
    print('test\n', test_dfs)
    # wwr.evaluate_final_dataset(test_dfs, val_dfs, run_path, names, cfg.split)


if __name__ == "__main__":
    main()


