import numpy as np
from numpy.core.fromnumeric import shape,reshape
from sigmoid import sigmoid

def backpropagation(AL, Y, caches, acache):
    """
    backpropagation of the model
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    #Y = Y.reshape(AL.shape)
    #TODO: temporary fixed the caches, still pendant a better fix with the A0 or X
    caches = np.array(caches, dtype=object)
    A_caches, W_caches, b_caches, Z_caches = caches.T
    
    for l in range(L):
        acache["a" + str(l + 1)] = A_caches[l]
        acache["w" + str(l + 1)] = W_caches[l]
        acache["b" + str(l + 1)] = b_caches[l]
        acache["z" + str(l + 1)] = Z_caches[l]

    #last layer derivative
    # temp dAL = - (np.divide(Y,AL) - np.divide( 1 - Y, 1 - AL))
    #current_cache = caches[-1]
    dZ_temp = AL - Y
    dW_temp = 1/m * np.dot(dZ_temp, acache["a" + str(L-1)].T)
    db_temp = 1/m * np.sum(dZ_temp,axis=1,keepdims=True)
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    for l in reversed(range(L-1)):
        
        #hidden layers derivatives
        dA_prev_temp = np.dot(W_caches[l+1].T, dZ_temp)
        one = np.ones(shape(Z_caches[l]))
        sigmoid_caches,temp = sigmoid(Z_caches[l])
        del temp
        dZ_temp = np.multiply(dA_prev_temp, np.multiply(sigmoid_caches, sigmoid_caches - 1))
        dW_temp = 1/m * np.dot(dZ_temp, acache["a" + str(l)].T)
        db_temp = 1/m * np.sum(dZ_temp,axis=1,keepdims=True)
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads