from update_params import update
import numpy as np
from init_params import init_params
from forward_propagation import fw_prop
from cost import cost_function
from backpropagation import backpropagation
from update_params import update
from data_prep import data_preparation
from numpy.core.fromnumeric import shape,reshape

def model(X, Y, layer_dims, learning_rate=0.0055, num_iters=5000, print_cost=False ):
    """
    unify and implement the model
    """
    np.random.seed(1)
    costs = []

    parameters = init_params(layer_dims)

    for i in range(0,num_iters):
        AL, caches = fw_prop(X, parameters)
        print("caches: " + str(shape(caches)))
        cost =  cost_function(AL,Y)
        grads = backpropagation(AL,Y,caches)
        parameters = update(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

dims = [784, 300, 100, 10]
X_train, y_train, X_valid, y_valid, X_test, y_test = data_preparation()
model(X_train, y_train, layer_dims=dims, num_iters=2500,print_cost=True)