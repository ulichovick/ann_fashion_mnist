import numpy as np

def softmax(z):
    """
    implement softmax function
    """
    AL = np.exp(z) / np.sum(np.exp(z), axis=0)
    return AL

scores = [3.0, 1.0, 0.2]
print(softmax(scores))