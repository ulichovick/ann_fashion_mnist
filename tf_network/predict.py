from .data_prep import data_prep
import numpy as np
from PIL import Image
import PIL.ImageOps 
import matplotlib.pyplot as plt
from tensorflow import keras

def predict(model,data):
    """
    make predictions
    """
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    file = data
    #open the image and convert it to black and white or greyscale
    fname = Image.open(file).convert('L')

    #invert the colors (the dataset is trained inverted, so the clothes are in white and the background in black)
    fname = PIL.ImageOps.invert(fname)
    
    height = 28
    width = 28
    fname = fname.resize((width,height))
    image = np.asarray(fname)

    my_image = image/255.
    my_image = my_image.reshape(28,28)
    my_image = my_image.reshape(1,-1)


    plt.imshow(image)
    plt.show()

    y_pred = model.predict(my_image)
    predictions = np.argmax(y_pred)
    print(predictions)
    print("The algorithm predicts: " + str(np.array(class_names)[predictions]))

#model = keras.models.load_model("ann_tf_fashmnist.h5")
#predict(model)