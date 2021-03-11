import numpy as np
from activation_neurons import activation
from softmax import softmax

def fw_prop(X, params):
    """
    forward propagation of the network
    """
    caches = []
    A = X
    L = len(params) //2
    for l in range(1,L):
        A_prev = A
        A, cache = activation(A_prev,
                            params['W' + str(l)],
                            params['b' + str(l)])
        caches.append(cache)

    A_prev = A
    A, cache = activation(A_prev,
                        params['W' + str(L)],
                        params['b' + str(L)])
    AL = softmax(A_prev)
    caches.append(cache)

    return AL, caches