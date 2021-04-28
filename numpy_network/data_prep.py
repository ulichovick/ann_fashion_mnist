from tensorflow import keras
import numpy as np
import sys
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

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
    #plt.imshow(x_train_full[0])
    #plt.title(class_names[y_train_full[0]])
    #plt.show()

    #normalize the data so that they are the same scale
    x_train_norm = x_train_full / 255
    x_test_norm = X_test / 255

    #split up the data in train and validation
    X_valid, X_train = x_train_norm[:5000], x_train_norm[5000:]
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = x_test_norm
    #plt.imshow(X_valid[0])
    #plt.title(class_names[y_valid[0]])
    #plt.show()
    #reshape the matrix data into a single vector 28 * 28 = 784 and put proper dimensions to (10,m) arrays
    #maybe needed for loss function idk xD
    #print(str(shape(y_train.reshape(1,-1))))
    #print("Train set shape " + str(y_valid[0]))
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    y_valid = keras.utils.to_categorical(y_valid)
    y_train = y_train.T
    y_test = y_test.T
    y_valid = y_valid.T
    
    #assert the shapes
    # print(y_train.shape)
    #plt.imshow(X_valid[1])
    #plt.title(class_names[np.argmax(y_valid[:,1])])
    #plt.show()

    X_valid = X_valid.reshape(X_valid.shape[0], -1).T
    X_train = X_train.reshape(X_train.shape[0], -1).T
    X_test = X_test.reshape(X_test.shape[0], -1).T
    #test = X_train[:,0]
    #test = test.reshape(len(X_train), 1)
    #print("Train set shape " + str(np.array(y_valid[:,0])))
    #print("Validation set shape" + str(X_valid.shape))
    #print("Test set shape" + str(X_test.shape))
    #print("train classes shape" + str(y_train.shape))
    return X_train, y_train, X_valid, y_valid, X_test, y_test
