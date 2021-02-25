from numpy.core.fromnumeric import shape
import numpy as np

def cost_function(AL, Y):
    """
    cost function
    """
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))

    return cost