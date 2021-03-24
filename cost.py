import numpy as np

def cost_function(AL, Y):
    """
    cost function
    """
    m = Y.shape[1]
    cost = -(1/m) * np.sum(Y *np.log(AL))

    return cost
