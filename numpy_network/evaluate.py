import os
import matplotlib
matplotlib.use('agg')
import json
from matplotlib import pyplot as plt
import numpy as np
from .forward_propagation import fw_prop
from sklearn.metrics import confusion_matrix, classification_report
import gc

def evaluate(params,data,y,data_scores):
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
    with open('numpy_network/'+data_scores+'.json', 'w') as file:
        json.dump(class_rep, file, indent=4)
    del caches, acache, AL, params, data, y
    gc.collect()
    return predictions, conf_mat, class_rep, class_names

def plot(conf_mat, class_names, data_set):
    """
    plot the confussion matrix
    """
    UPLOAD_FOLDER = 'static/graphs'
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
    file = os.path.join(UPLOAD_FOLDER,str(data_set+'.png'))
    plt.savefig(file)
    plt.clf()
    plt.close('all')
    del conf_mat, class_names
    gc.collect()
    return file
