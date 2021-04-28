import matplotlib.pyplot as plt
from tensorflow import keras

def data_prep(X_train, Y_train, X_test, Y_test):
    """
    import dataset, split and normalize it
    """

    #import the data
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    #print(X_train[0])
    #check the data
    #plt.imshow(X_train[10])
    #plt.title(class_names[Y_train[10]])
    #plt.show()
    #print(X_train[10])

    #normalize the data
    X_train = X_train / 255
    X_test = X_test / 255

    #split the data into train/validation/test
    X_valid, X_train = X_train[:5000], X_train[5000:]
    Y_valid, Y_train = Y_train[:5000], Y_train[5000:]
    X_test = X_test
    
    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test, class_names
