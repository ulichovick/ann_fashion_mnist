from .data_prep import data_prep
from .seq_api import seq_api
from .compile import compile
from .evaluate import evaluate
from tensorflow import keras

def model():
    """
    unify de model
    """
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train, Y_train), (X_test,Y_test) = fashion_mnist.load_data()
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test, class_names = data_prep(X_train, Y_train, X_test, Y_test)
    model = seq_api()
    model = compile(model, X_train, Y_train, X_valid, Y_valid)
    model.save("ann_tf_fashmnist.h5")

model()