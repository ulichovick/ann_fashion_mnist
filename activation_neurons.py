import numpy as np
from hypothesis import hypothesis
from sigmoid import sigmoid

def activation(A_prev, W, b):
    """
    unifies the hypothesis and the softmax activation
    """
    Z, Z_cache = hypothesis(A_prev,W,b)
    A, A_cache = sigmoid(Z)

    cache = (Z_cache, A_cache)

    return A, cache
