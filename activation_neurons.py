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
    print("activation cache " + str(shape(A_cache)))
    print("W cache " + str(shape(W_cache)))
    print("b cache " + str(shape(b_cache)))
    print("Z cache " + str(shape(Z_cache)))
    
    print("activation " + str(shape(A)))
    print("W " + str(shape(W)))
    print("b " + str(shape(b)))
    print("Z " + str(shape(Z)))
    cache = (A_cache, W_cache, b_cache, Z_cache)

    return A, cache
