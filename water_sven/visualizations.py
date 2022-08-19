import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import pickle
import water_w_regression as wwr
import config as cfg


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
        print('history2', history)
        print(history.history.items())
    for key, metric in history.history.items():
        key_n = key.replace('_', ' ').title()
        if key[:3] != 'val':
            plt.figure()
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
            plt.grid(which='both', axis='y')
            #plt.show()
            #Save
            plt.savefig(os.path.join(save_path, key))
            plt.close()
            #asd


def histogram(df, stat, title, path, file_n, limit_above=120):
    plt.figure()
    if not type(max(df)) == str and max(df) >= limit_above:
        bins = list(range(-4, 101, 5))
        ax = sns.displot(df, stat=stat, bins=bins)
    else:
        ax = sns.displot(df, stat=stat)
    if type(max(df)) == str:
        ax.set_xticklabels(rotation=90)
        plt.xlabel('')
    plt.title(title)
    # plt.show(bbox_inches='tight')
    plt.savefig(path + file_n, bbox_inches='tight')
    plt.close()


# plot as scatter plot and compute a regression line, including alpha and beta
def scatterplotRegression(df, run_path, file_name=''):
    """Plots a Regressionplot with a text annotation containing additional information
    Args:
        df (pd.DataFrame): DataFrame consisting of the Actual and the Predicted Volatility Values
        model_type (str): Name of the model
        EPOCHS (int): the amount of epochs, the model was trained on
        BATCH_SIZE (int): number of training examples
        It does not return anything, but plots a regression plot and saves it
    """
    beta, alpha = np.polyfit(df.Actual, df.Prediction, 1)
    corr_value = df.corr(method="pearson")
    pearson_corr = corr_value["Prediction"][1]
    print(pearson_corr)
    parameter_text = "\n".join(
        (
            r"$pearson\ corr=%.2f$" % (pearson_corr,),
            r"$slope: \beta=%.2f$" % (beta,),
            r"$intercept: \alpha=%.2f$" % (alpha,),
        )
    )
    print(parameter_text)
    plt.figure()
    sns.set(rc={"figure.figsize": (15, 10)})
    sns.regplot(
        data=df,
        x="Actual",
        y="Prediction",
        x_ci="sd",
        scatter=True,
        line_kws={"color": "red"},
    )
    # add text annotation
    # set the textbox color to purple
    purple_rgb = (255.0 / 255.0, 2.0 / 255.0, 255.0 / 255.0)

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
    plt.title('Scatterplot', fontsize=14)
    plt.xlabel('Actual', fontsize=14)
    plt.ylabel('Prediction', fontsize=14)
    # finally save the chart to disk
    plt.savefig(run_path + 'Scatterplot_' + file_name + ".png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    # print a few regression statistics
    print("Beta of ", df.Prediction.name, " =", round(beta, 4))
    print("Alpha of ", df.Prediction.name, " =", round(alpha, 4))


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
        print('Creating CM', i)
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
            print("Normalized confusion matrix")
        else:
            thresh = thresh * len(label_true)
            print('Confusion matrix, without normalization')

        print('threshold for CM', thresh)
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
        print('Creating CM', i)
        plt.figure()
        cm = confusion_matrix(y_true=label_true, y_pred=prediction_true)

        if i == 'normalized':
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.around(cm, decimals=2)
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

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
    # run_path = r'/mnt/datadisk/data/Projects/water/trainHistory_aug/source_of_drinking_water_(categorized_by_type)__max/vgg19_wimagenet_unfl100_d0_lr0.0001_momentum0.9_optimizerSGD_horizontal_flip_featurewise_8/'
    # with open(run_path + '/pickle_cm', 'rb') as f:
    #     [test_true, test_prediction, label_mapping] = pickle.load(f)
    #
    # # Confusion matrix
    # cm_plot_labels = list(label_mapping.values())
    # plot_CM(test_true, test_prediction, cm_plot_labels, run_path +
    #                        'ConfusionMatrix')
    # plot_CM_sns(test_true, test_prediction, cm_plot_labels, run_path +
    #                           'ConfusionMatrix')
    # run_path = '/mnt/datadisk/data/Projects/water/trainHistory_aug_testing//time_to_get_to_water_source_+_penalty_capped/vgg19_wimagenet_unfl100_d0_lr0.0001_momentum0.9_optimizerSGD_horizontal_flip_featurewise_3/'
    # with open(run_path + 'dfs_in', 'rb') as f:
    #     [labels_df, train_df, validation_df, test_df] = pickle.load(f)
    # label_name = 'time to get to water source + penalty (capped)'
    # for df_n, df in zip(['All', 'Train', 'Validation', 'Test'],
    #                     [labels_df, train_df, validation_df, test_df]):
    #     for col in df.columns:
    #         if col in [label_name, 'transformed', 'normalized', 'label']:
    #             add_n = col
    #             if col == label_name:
    #                 add_n = 'Raw Data'
    #             elif col == 'label':
    #                 add_n = 'Input'
    #             title = label_name.replace('_', ' ').title() + ' ' + \
    #                     '(n=' + str(len(df)) + ')'
    #             histogram(df[col], 'probability', title, run_path,
    #                                      label_name + df_n + ' ' + add_n)
    # run_path = '/mnt/datadisk/data/Projects/water/trainHistory_aug_testing//time_to_get_to_water_source_+_penalty_capped/vgg19_wimagenet_unfl100_d0_lr0.0001_momentum0.9_optimizerSGD_horizontal_flip_featurewise_17/'
    # with open(run_path + '/pickle_prediction_true_1', 'rb') as f:
    #     [test_pred, test_true_list, label_mapping, add_params, run_path, labels_df] = pickle.load(f)
    # print('test pred', test_pred)
    # print('test true', test_true_list)
    # print(add_params)
    # test_prediction = pd.DataFrame({'Prediction': np.array(test_pred).reshape(-1),
    #                                 'Actual': np.array(test_true_list).reshape(-1)})
    # test_prediction_before_detransform = copy.deepcopy(test_prediction)
    # print('test pred df', test_prediction)
    # with open(run_path + '/pickle_prediction_true_norm', 'wb') as f:
    #     pickle.dump([test_prediction, label_mapping, add_params, run_path], f)
    # if cfg.label_transform or cfg.label_normalization:
    #     test_prediction = wwr.reverse_norm_transform(test_prediction,
    #                                              cfg.label_normalization, cfg.label_transform,
    #                                              additional_params=add_params, run_path=run_path)
    # print('test pred', test_prediction)
    # with open(run_path + '/pickle_prediction_true', 'wb') as f:
    #     pickle.dump([test_prediction, label_mapping, add_params, run_path], f)
    #
    # for df_n, df in zip(['Test prediction', 'Test True'],
    #                     [test_prediction['Prediction'], test_prediction['Actual']]):
    #     title = 'asf'.replace('_', ' ').title() + ' ' + df_n + ' ' + \
    #             '(n=' + str(len(df)) + ')'
    #     histogram(df, 'probability', title, run_path, title)
    #
    # print(test_prediction)
    # for name, df in zip(['normalized', 'denormalized'], [test_prediction_before_detransform, test_prediction]):
    #     scatterplotRegression(df=df, run_path=run_path, file_name=name)
    # run_path = '/mnt/datadisk/data/Projects/water/trainHistory_aug_cat//source_of_drinking_water_categorized_piped__max/vgg19_wimagenet_unfl100_d0_lr0.0001_momentum0.9_optimizerSGD_horizontal_flip_featurewise_3/'
    # with open(run_path + '/pickle_prediction_true_1', 'rb') as f:
    #     [test_pred, test_true_list, label_mapping, add_params, run_path, labels_df] = pickle.load(f)
    # print(len(test_pred), len(test_true_list))
    # print('testpred', test_pred)
    # print('true', test_true_list)
    # test_prediction = np.array(test_pred)
    # test_true = np.array(test_true_list)
    # # add_params['f1 micro'] = f1_score(test_true, test_prediction, average='micro')
    #
    # # Confusion matrix
    # cm_plot_labels = list(label_mapping.values())
    # plot_CM(test_true, test_prediction, cm_plot_labels, run_path +
    #                        'ConfusionMatrix')
    # run_path = '/mnt/datadisk/data/Projects/water/trainHistory_final_pca_dist/PCA_w_weighting_urban0/76r2_65b_vgg19_wimagenet_unfl100_d0_lr0.0001_momentum0.9_optimizerSGD_horizontal_flip_vertical_flip_featurewise_1/'
    run_path = '/mnt/datadisk/data/Projects/water/trainHistory_distance_normed_transformed_dropped/time_to_get_to_water_source_refurbished/56r2_vgg19_wimagenet_unfl100_d0_lr0.0001_momentum0.9_optimizerSGD_shear_0.2zoom_0.2horizontal_flip_featurewise_1/'

    with open(run_path + '/pickle_prediction_true', 'rb') as f:
        [test_prediction, label_mapping, add_params, run_path2] = pickle.load(f)
    print(test_prediction)
    scatterplotRegression(df=test_prediction, run_path=run_path, file_name='Original Data')


if __name__ == "__main__":
    main()


