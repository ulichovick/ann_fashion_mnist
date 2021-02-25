import numpy as np
from numpy.core.fromnumeric import shape,reshape

def backpropagation(AL, Y, caches, cache):
    """
    backpropagation of the model
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    linear_cache, activation_cache = cache
    #last layer derivative
    dAL = - (np.divide(Y,AL) - np.divide( 1 - Y, 1 - AL))
    current_cache = caches[-1]
    dZ_temp = np.multiply(dAL, 1 - np.power(activation_cache, 2))
    dA_prev_temp = np.dot(cache[1].T, dZ_temp)
    dW_temp = 1/m * np.dot(dZ_temp, cache[0].T)
    db_temp = 1/m * np.sum(dZ_temp,axis=1,keepdims=True)
    grads["dA" + str(L)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        #hidden layers derivatives
        dA_prev_temp = np.dot(cache[1].T, dZ_temp)
        dZ_temp = np.multiply(dAL, 1 - np.power(activation_cache, 2))
        dW_temp = 1/m * np.dot(dZ_temp, cache[0].T)
        db_temp = 1/m * np.sum(dZ_temp,axis=1,keepdims=True)
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp