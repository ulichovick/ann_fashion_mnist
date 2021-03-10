from data_prep import data_prep
from seq_api import seq_api
from compile import compile
from evaluate import evaluate
from predict import predict
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
    evaluate(model, X_test, Y_test)
    
    predict(model, class_names)
    print(X_train[0].shape)

model()