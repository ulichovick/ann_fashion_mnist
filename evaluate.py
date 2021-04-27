from activation_neurons import activation
import numpy as np
from PIL import Image
import PIL.ImageOps 
from data_prep import data_preparation

def evaluate(params,data,y):
    """
    try to predict
    """
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    #open the image and convert it to black and white or greyscale
    fname = Image.open('camiseta.jpg').convert('L')

    #invert the colors (the dataset is trained inverted, so the clothes are in white and the background in black)
    fname = PIL.ImageOps.invert(fname)
    
    fname.save('graysc.png')
    height = 28
    width = 28
    fname = fname.resize((width,height))
    image = np.asarray(fname)

    my_image = image/255.
    my_image = my_image.reshape(28,28)
    my_image = my_image.reshape(1,-1)

    caches = []
    A = data

    L = len(params) //2
    for l in range(1,L):
        A_prev = A
        A, cache = activation(A_prev,
                            params['W' + str(l)],
                            params['b' + str(l)],
                            activation="sigmoid")

    AL, cache = activation(A,
                        params['W' + str(L)],
                        params['b' + str(L)], 
                        activation="softmax")

    predictions = np.argmax(AL, axis=0)

    caches.append(predictions)

    return predictions


X_train, y_train, X_valid, y_valid, X_test, y_test = data_preparation()

parameters = np.load('parameters.npy', allow_pickle=True)[()]

Y_prediction_train = evaluate(parameters,X_train,y_train)
y_train = np.argmax(y_train, axis=0)
print("train accuracy: {} %".format(np.mean(Y_prediction_train == y_train) * 100))

Y_prediction_valid = evaluate(parameters,X_valid,y_valid)
y_valid = np.argmax(y_valid, axis=0)
print("validation accuracy: {} %".format(np.mean(Y_prediction_valid == y_valid) * 100))

Y_prediction_test = evaluate(parameters,X_test,y_test)
y_test = np.argmax(y_test, axis=0)
print("test accuracy: {} %".format(np.mean(Y_prediction_test == y_test) * 100))
