import numpy as np

def hypothesis(A, W, b):
    """
    hypothesis 
    """
    Z = np.dot(W,A)+b

    cache = (A,W,b)
    return Z, cache