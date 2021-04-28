import numpy as np

def drelu(z):
    """
    relu derivative
    """
    r = 1 * (z >= 0)
    return r