from tensorflow import keras

import gc

def data_preparation():
    """
    this function extracts the dataset
    and splits the train and test set
    """   
    fashion_mnist = keras.datasets.fashion_mnist
    (x_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

    x_train_norm = x_train_full / 255
    x_test_norm = X_test / 255


    X_valid, X_train = x_train_norm[:5000], x_train_norm[5000:]
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = x_test_norm

    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    y_valid = keras.utils.to_categorical(y_valid)
    y_train = y_train.T
    y_test = y_test.T
    y_valid = y_valid.T

    X_valid = X_valid.reshape(X_valid.shape[0], -1).T
    X_train = X_train.reshape(X_train.shape[0], -1).T
    X_test = X_test.reshape(X_test.shape[0], -1).T

    del x_test_norm, x_train_norm, x_train_full, y_train_full
    gc.collect()
    return X_train, y_train, X_valid, y_valid, X_test, y_test
