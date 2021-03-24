from update_params import update
import numpy as np
from init_params import init_params
from forward_propagation import fw_prop
from cost import cost_function
from backpropagation import backpropagation
from update_params import update
from data_prep import data_preparation
from predict import predict
import matplotlib.pyplot as plt
import time

def model(X, Y, layer_dims, learning_rate=0.01, num_iters=5, print_cost=False ):
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

        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

        print ("Cost after iteration %i: %f" % (i, cost))

        costs.append(cost)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    #plt.show()
    Y_prediction_train = predict(parameters,X)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y)) * 100))


start_time = time.time()
dims = [784, 100, 100, 10]
X_train, y_train, X_valid, y_valid, X_test, y_test = data_preparation()
model(X_train, y_train, layer_dims=dims, num_iters=500,print_cost=True)
print("--- %s seconds ---" % round(time.time() - start_time, 2))