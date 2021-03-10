import numpy as np
import tensorflow as tf
from tensorflow import keras

def seq_api():
    """
    define the structure of the model and train it
    """
    #nice
    np.random.seed(69)
    tf.random.set_seed(69)

    #define the model
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28,28)))
    model.add(keras.layers.Dense(300, activation="sigmoid"))
    model.add(keras.layers.Dense(100, activation="sigmoid"))
    model.add(keras.layers.Dense(10, activation="softmax"))

    #check the model arq
    model.summary()
    keras.utils.plot_model(model)

    return model
