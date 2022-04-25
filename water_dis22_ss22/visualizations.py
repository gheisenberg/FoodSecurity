import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def annot_max(x, y, ax=None):
    """
    Annotate maximum of graph and last point of graph

    Args:
        x (np array): X values of graph
        y (np array): Y values of graph

    Returns:
        None
    """
    annotations = [(x[np.argmax(y)], max(y)), (x[-1], y[-1])]
    for (xmax, ymax) in annotations:
        text = "{:.2f}".format(ymax)
        if not ax:
            ax=plt.gca()
        ax.annotate(text, xy=(xmax, ymax))

def plot_history(history, save_path, verbose=1):
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
        if key[:3] != 'val':
            plt.figure()
            plt.plot(history.history[key])
            if key != 'lr':
                plt.plot(history.history['val_' + key])
                annot_max(np.arange(len(history.history['val_' + key])), history.history['val_' + key])
            plt.title('model ' + key)
            plt.ylabel(key)
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            if key == 'loss':
                plt.ylim(0, 2)
            elif key == 'accuracy':
                plt.ylim(0, 1)
            plt.grid(which='both', axis='y')
            #plt.show()
            #Save
            plt.savefig(os.path.join(save_path, key))
            #asd


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
        cm = confusion_matrix(y_true=label_true, y_pred=prediction_true)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if i == 'normalized':
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.around(cm, decimals=2)
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        thresh = 0.7
        print('threshold for CM', thresh)
        for k, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, k, cm[k, j],
                horizontalalignment="center",
                color="black")# if cm[k, j] > thresh else "black")
        plt.colorbar()
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(save_path + i)


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
