import numpy as np

def sigmoid(z):
    """
    activation of the hidden layer neurons
    """
    s = 1/(1 + np.exp(-z))
    
    return s, z

# sanity check
# print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))