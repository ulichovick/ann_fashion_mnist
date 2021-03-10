import numpy as np
from PIL import Image
import PIL.ImageOps 
import scipy.misc
import matplotlib.pyplot as plt

def predict(model, class_names):
    """
    make predictions
    """
    #open the image and convert it to black and white or greyscale
    fname = Image.open('camiseta.jpg').convert('L')

    #invert the colors (the dataset is trained inverted, so the clothes are in white and the background in black)
    fname = PIL.ImageOps.invert(fname)
    
    fname.save('graysc.png')
    height = 28
    width = 28
    fname = fname.resize((width,height))
    image = np.asarray(fname)
    print(image.shape)
    my_image = image/255.
    my_image = my_image.reshape(28,28)
    my_image = my_image.reshape(1,-1)
    print(my_image.shape)

    plt.imshow(image)
    plt.show()

    y_pred = model.predict(my_image)
    predictions = np.argmax(y_pred)
    print(predictions)
    print("The algorithm predicts: " + str(np.array(class_names)[predictions]))
