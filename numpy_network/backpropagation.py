import numpy as np

def backpropagation(AL, Y, caches, acache):
    """
    backpropagation of the model
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    caches = np.array(caches, dtype=object)
    A_caches, W_caches, b_caches, Z_caches = caches.T
    
    for l in range(L):
        acache["a" + str(l + 1)] = A_caches[l]

    #last layer derivative
    dZ_temp = AL - Y
    dW_temp = 1/m * np.dot(dZ_temp, acache["a" + str(L - 1)].T)
    db_temp = 1/m * np.sum(dZ_temp,axis=1,keepdims=True)
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    #hidden layer derivatives
    for l in reversed(range(L-1)):

        #hidden layers derivatives
        dA_prev_temp = np.dot(W_caches[l+1].T, dZ_temp)
        dZ_temp = np.multiply(dA_prev_temp, np.multiply(acache["a" + str(l+1)],1 - acache["a" + str(l+1)]))
        dW_temp = 1/m * np.dot(dZ_temp, acache["a" + str(l)].T)
        db_temp = 1/m * np.sum(dZ_temp,axis=1,keepdims=True)
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    del caches, A_caches, Z_caches, W_caches, b_caches
    return grads