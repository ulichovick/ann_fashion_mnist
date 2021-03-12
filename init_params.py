import numpy as np

def init_params(L_dims):
    """
    initialize hyper parameters w and b 
    """
    np.random.seed(3)
    params = {}
    L = len(L_dims)

    for l in range(1,L):
        params ['W' + str(l)] = np.random.rand(L_dims[l],L_dims[l-1]) * 0.01
        params ["b" + str(l)] = np.zeros((L_dims[l], 1))

    return params

#sanity check
#parameters = init_params([5,3,3])
#print("parameters  = " + str(parameters))
