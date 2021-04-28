
def relu(z):
    """
    relu
    """
    r = z * (z > 0)
    return r, z