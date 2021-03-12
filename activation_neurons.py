import numpy as np
from hypothesis import hypothesis
from sigmoid import sigmoid
from numpy.core.fromnumeric import reshape, shape, size

def activation(A_prev, W, b):
    """
    unifies the hypothesis and the softmax activation
    """
    Z, W_cache, b_cache = hypothesis(A_prev,W,b)
    A, Z_cache = sigmoid(Z)
    A_cache = A
    cache = (A_cache, W_cache, b_cache, Z_cache)

    return A, cache
