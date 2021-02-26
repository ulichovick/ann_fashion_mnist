import numpy as np
from numpy.core.fromnumeric import shape,reshape

def update(parameters, grads, learning_rate):
    """
    update parameters with respective derivatives
    """
    L = len(parameters)

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate + grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate + grads["db" + str(l+1)]
    
    return parameters