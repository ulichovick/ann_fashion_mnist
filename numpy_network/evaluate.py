import numpy as np
from .forward_propagation import fw_prop

def evaluate(params,data,y):
    """
    try to predict
    """
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    AL, caches, acache = fw_prop(data, params)

    predictions = np.argmax(AL, axis=0)

    caches.append(predictions)

    return predictions
