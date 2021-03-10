import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def compile(model, X_train, Y_train, X_valid, Y_valid):
    """
    compile the model
    """
    model.compile(loss="sparse_categorical_crossentropy",
                    optimizer="sgd",
                    metrics=["accuracy"])
    model_history = model.fit(X_train, Y_train, epochs=30,
                                validation_data = (X_valid,Y_valid))
    print(model_history.params)
    pd.DataFrame(model_history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
    
    return model