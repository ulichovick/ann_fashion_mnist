from activation_neurons import activation
import numpy as np
from PIL import Image
import PIL.ImageOps 
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape

def predict(params,data,y):
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
    #print(image.shape)
    my_image = image/255.
    my_image = my_image.reshape(28,28)
    my_image = my_image.reshape(1,-1)
    #print(my_image.shape)

    #plt.imshow(image)
    #plt.show()

    caches = []
    A = data
    #print(A.shape)

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
    #print(shape(AL))
    predictions = np.argmax(AL, axis=0)
    #print(shape(predictions))
    caches.append(predictions)
    #print(AL)
    #print(y[:,0])
    #print("The algorithm predicts: " + str(np.array(class_names)[predictions]))
    clas = np.argmax(y[:,0])
    #print("The actual class: " + str(np.array(class_names)[clas]))
    return predictions