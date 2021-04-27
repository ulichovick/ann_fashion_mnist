from activation_neurons import activation
import numpy as np
from PIL import Image
import PIL.ImageOps 
import matplotlib.pyplot as plt

def predict(params,data):
    """
    try to predict
    """
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    #open the image and convert it to black and white or greyscale
    file = data
    fname = Image.open(file).convert('L')

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

    plt.imshow(image)
    plt.show()

    A = my_image.T

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
    del cache
    print("The network predicts: " + str(np.array(class_names)[predictions]))

    return predictions

parameters = np.load('parameters.npy', allow_pickle=True)[()]

Y_prediction_valid = predict(parameters,"camisa.jpg")