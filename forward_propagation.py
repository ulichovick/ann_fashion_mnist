import numpy as np
from activation_neurons import activation

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
    AL, cache = activation(A_prev,
                            params['W' + str(l)],
                            params['b' + str(l)])
    caches.append(cache)

    return AL, caches