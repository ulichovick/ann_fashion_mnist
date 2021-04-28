import numpy as np

def sigmoid(z):
    """
    activation of the hidden layer neurons
    """
    s = 1/(1 + np.exp(-z))
    
    return s, z
