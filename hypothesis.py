import numpy as np

def hypothesis(A, W, b):
    """
    hypothesis 
    """
    Z = np.dot(A,W.T)+b

    cache = (A,W,b)
    return Z, cache