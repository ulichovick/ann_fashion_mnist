import numpy as np

def cost_function(AL, Y):
    """
    cost function
    """
    m = Y.shape[1]
    L_sum = np.sum(np.multiply(Y,np.log(AL)))
    cost = -(1/m) * L_sum
    cost = np.squeeze(cost)  

    return cost
