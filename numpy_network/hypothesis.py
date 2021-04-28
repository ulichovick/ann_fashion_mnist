import numpy as np

def hypothesis(A, W, b):
    """
    hypothesis 
    """
    Z = np.dot(W,A)+b

    return Z, W, b