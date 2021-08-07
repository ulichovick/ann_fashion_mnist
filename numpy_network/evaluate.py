import matplotlib
from sklearn.utils.multiclass import class_distribution
matplotlib.use('agg')
import base64
from io import BytesIO
from matplotlib import pyplot as plt
import numpy as np
from .forward_propagation import fw_prop
from sklearn.metrics import confusion_matrix, classification_report
import gc

def evaluate(params,data,y):
    """
    try to predict
    """
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    AL, caches, acache = fw_prop(data, params)

    predictions = np.argmax(AL, axis=0)

    y_fixed = np.argmax(y, axis=0)
    conf_mat = confusion_matrix(y_fixed, predictions)
    class_rep = classification_report(y_fixed, predictions, output_dict=True, target_names=class_names)
    del caches, acache, AL, params, data, y
    gc.collect()
    return predictions, conf_mat,class_rep, class_names

def plot(conf_mat, class_names):
    """
    plot the confussion matrix
    """
    plt.imshow(conf_mat)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = plt.text(j, i, conf_mat[i, j],
                            ha="center", va="center", color="w")
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, labels=class_names, rotation=-30)
    plt.yticks(tick_marks, labels=class_names)
    plt.tick_params(top=True, bottom=False,
                    labeltop=True, labelbottom=False)
    plt.ylabel("True Values")
    plt.xlabel("Predicted Values")
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.clf()
    plt.close('all')
    del conf_mat, class_names
    gc.collect()
    plot_url = base64.b64encode(img.getbuffer()).decode("ascii")
    del img
    return plot_url
