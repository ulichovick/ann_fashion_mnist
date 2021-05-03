from .data_prep import data_prep
from tensorflow import keras

def evaluate(model, X_test, Y_test):
    """
    evaluate the model
    """
    
    print(model.evaluate(X_test, Y_test))


model = keras.models.load_model("ann_tf_fashmnist.h5")
fashion_mnist = keras.datasets.fashion_mnist
(X_train, Y_train), (X_test,Y_test) = fashion_mnist.load_data()
X_train, X_valid, X_test, Y_train, Y_valid, Y_test, class_names = data_prep(X_train, Y_train, X_test, Y_test)
evaluate(model, X_test, Y_test)