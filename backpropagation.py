import numpy as np
from numpy.core.fromnumeric import shape


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

    #last layer derivative
    dZ_temp = AL - Y
    dW_temp = 1/m * np.dot(dZ_temp, A_caches[L-1].T)
    db_temp = 1/m * np.sum(dZ_temp,axis=1,keepdims=True)
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp
    print(shape(acache["a0"]))
    print(shape(A_caches[0]))
    #hidden layer derivatives
    for l in reversed(range(L-1)):
        
        #hidden layers derivatives
        dA_prev_temp = np.dot(W_caches[l+1].T, dZ_temp)
        
        dZ_temp = np.multiply(dA_prev_temp, np.multiply(A_caches[l],1 - A_caches[l]))
        dW_temp = 1/m * np.dot(dZ_temp, A_caches[l-1].T)
        db_temp = 1/m * np.sum(dZ_temp,axis=1,keepdims=True)
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        print("dw" + str(shape(grads["dW" + str(l + 1)])))
        print("db" + str(shape(grads["db" + str(l + 1)])))
    del caches, A_caches, Z_caches, W_caches, b_caches
    return grads