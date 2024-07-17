from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def accuracy(out, yb):
    preds = np.argmax(out, axis=1)
    return np.mean(preds == yb)


def plot_confusion_matrix(out, yb, title=None, path_to_save=None):
    preds = np.argmax(out, axis=1)
    cf_matrix = confusion_matrix(yb, preds).astype(np.int)
    plt.cla(), plt.clf()
    fig = sns.heatmap(cf_matrix, annot=True, cbar=False, fmt="d").get_figure()
    plt.ylabel('ground truth')
    plt.xlabel('predicted')
    if title is not None:
        plt.title(title)
    if path_to_save is not None:
        fig.savefig(path_to_save)
    return fig
