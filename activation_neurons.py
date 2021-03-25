import numpy as np
from hypothesis import hypothesis
from sigmoid import sigmoid
from softmax import softmax

def activation(A_prev, W, b, activation = ""):
    """
    unifies the hypothesis and the activation functions
    """
    if activation == "sigmoid":
        Z, W_cache, b_cache = hypothesis(A_prev,W,b)
        A, Z_cache = sigmoid(Z)
        A_cache = A
    elif activation == "softmax":
        Z, W_cache, b_cache = hypothesis(A_prev,W,b)
        A, Z_cache = softmax(Z)
        A_cache = A
    cache = (A_cache, W_cache, b_cache, Z_cache)

    return A, cache
