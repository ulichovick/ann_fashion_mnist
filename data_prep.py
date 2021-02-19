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
    (x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()

    #create array with class names
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    #show an image example of the data with their class name
    print(x_train_full[0])
    plt.imshow(x_train_full[0])
    plt.title(class_names[y_train_full[0]])
    plt.show()

    #normalize the data so that they are the same scale
    x_train_norm = x_train_full / 255
    x_test_norm = x_train_full / 255
    print(x_train_norm[0])
    X_valid, X_train = x_train_norm[:5000], x_train_norm[5000:]
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    plt.imshow(X_train[0])
    plt.title(class_names[y_train[0]])
    plt.show()


#trace memory
tracemalloc.start()

#call the func
data_preparation()

#display the memory usage
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("[ Top 10 ]")
for stat in top_stats[:10]:
    print(stat)