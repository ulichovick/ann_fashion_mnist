import numpy as np

def softmax(z):
    """
    implement softmax function
    """
    AL = np.exp(z) / np.sum(np.exp(z), axis=0)
    return AL
