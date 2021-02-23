from data_prep import data_preparation
from tensorflow import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tracemalloc
from data_prep import data_preparation
def sigmoid(z):
    """
    activation of the hidden layer neurons
    """
    s = 1/(1 + np.exp(-z))
    
    return s

# print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))