from activation_neurons import activation

def fw_prop(X, params):
    """
    forward propagation of the network
    """
    acache = {}
    caches = []
    A = X
    acache["a0"] = X
    L = len(params) //2
    for l in range(1,L):
        A_prev = A
        A, cache = activation(A_prev,
                            params['W' + str(l)],
                            params['b' + str(l)],
                            activation="sigmoid")
        caches.append(cache)

    AL, cache = activation(A,
                        params['W' + str(L)],
                        params['b' + str(L)], 
                        activation="softmax")
    caches.append(cache)

    return AL, caches, acache