import numpy as np
from numpy.core.fromnumeric import shape,reshape

def backpropagation(AL, Y, caches):
    """
    backpropagation of the model
    """
    grads = {}
    L = len(caches)
    print("layers" + str(L))
    m = AL.shape[1]
    #Y = Y.reshape(AL.shape)

    A_caches, W_caches, b_caches, Z_caches  = caches
    A_cache = A_caches[0][0]
    W_cache  = W_caches[0][1]
    b_cache  = b_caches[0][2]
    Z_cache  = Z_caches[1][0]
    #print(W_caches)
    print("z " + str(shape(Z_cache.T)))
    print("z data " + str(Z_cache.T))
    #print(" z " + str((b_cache[1][0])))

    #last layer derivative
    # temp dAL = - (np.divide(Y,AL) - np.divide( 1 - Y, 1 - AL))
    #current_cache = caches[-1]
    dZ_temp = AL - Y
    dA_prev_temp = np.dot(W_cache[1][L].T, dZ_temp)
    dW_temp = 1/m * np.dot(dZ_temp, A_cache[1][L].T)
    db_temp = 1/m * np.sum(dZ_temp,axis=1,keepdims=True)
    grads["dA" + str(L)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        #hidden layers derivatives
        dA_prev_temp = np.dot(W_cache[1][l].T, dZ_temp)
        dZ_temp = np.multiply(dA_prev_temp, 1 - np.power(Z_cache, 2))
        dW_temp = 1/m * np.dot(dZ_temp, A_cache[1][l].T)
        db_temp = 1/m * np.sum(dZ_temp,axis=1,keepdims=True)
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    
    return grads