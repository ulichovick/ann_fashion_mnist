from update_params import update
import numpy as np
from init_params import init_params
from forward_propagation import fw_prop
from cost import cost_function
from backpropagation import backpropagation
from update_params import update
from data_prep import data_preparation
from numpy.core.fromnumeric import shape,reshape
import matplotlib.pyplot as plt

def model(X, Y, layer_dims, learning_rate=0.0055, num_iters=5, print_cost=False ):
    """
    unify and implement the model
    """
    np.random.seed(1)
    costs = []

    parameters = init_params(layer_dims)

    for i in range(0,num_iters):
        AL, caches, acache = fw_prop(X, parameters)
        cost =  cost_function(AL,Y)
        grads = backpropagation(AL,Y,caches, acache)
        parameters = update(parameters, grads, learning_rate)


        print ("Cost after iteration %i: %f" % (i, cost))

        costs.append(cost)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

dims = [784, 50, 50, 10]
X_train, y_train, X_valid, y_valid, X_test, y_test = data_preparation()
model(X_train, y_train, layer_dims=dims, num_iters=200,print_cost=True)