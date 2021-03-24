import numpy as np

def update(parameters, grads, learning_rate):
    """
    update parameters with respective derivatives
    """
    L = len(parameters) // 2

    for l in range(L):
        #print("W: " + str(parameters["W" + str(l+1)]))
        #print("b: " + str(parameters["b" + str(l+1)]))
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        #print("W despues: " + str(parameters["W" + str(l+1)]))
        #print("b despues: " + str(parameters["b" + str(l+1)]))
    return parameters