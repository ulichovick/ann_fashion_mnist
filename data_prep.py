from numpy.core.fromnumeric import reshape, shape, size
from tensorflow import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tracemalloc
def data_preparation():
    """
    this function extracts the dataset
    and splits the train and test set
    """
    #extract the data and split test/train    
    fashion_mnist = keras.datasets.fashion_mnist
    (x_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

    #create array with class names
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    #show an image example of the data with their class name
    plt.imshow(x_train_full[0])
    plt.title(class_names[y_train_full[0]])
    plt.show()

    #normalize the data so that they are the same scale
    x_train_norm = x_train_full / 255
    x_test_norm = X_test / 255

    #split up the data in train and validation
    X_valid, X_train = x_train_norm[:5000], x_train_norm[5000:]
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = x_test_norm

    #reshape the matrix data into a single vector 28 * 28 = 784 and put proper dimensions to (x,) arrays
    #maybe needed for loss function idk xD
    print(str(shape(y_train.reshape(1,-1))))
    y_train = y_train.reshape(1,-1)
    y_test = y_test.reshape(1,-1)
    y_valid = y_valid.reshape(1,-1)
    print(y_train)
    plt.imshow(X_train[1])
    plt.title(class_names[y_train[0][1]])
    plt.show()

    X_valid = X_valid.reshape(X_valid.shape[0], -1).T
    X_train = X_train.reshape(X_train.shape[0], -1).T
    X_test = X_test.reshape(X_test.shape[0], -1).T
    print("Train set shape" + str(X_train.shape))
    print("Validation set shape" + str(X_valid.shape))
    print("Test set shape" + str(X_test.shape))
    print("train classes shape" + str(y_train.shape))
    return X_train, y_train, X_valid, y_valid, X_test, y_test

#trace memory
#tracemalloc.start()

data_preparation()
#display the memory usage
#snapshot = tracemalloc.take_snapshot()
#top_stats = snapshot.statistics('lineno')

#print("[ Top 10 ]")
#for stat in top_stats[:10]:
#    print(stat)