from flask import Flask
from flask.templating import render_template
#from numpy_network.predict import predict as np_pred
#from tf_network.predict import predict as tf_pred
#import numpy as np
#from tensorflow import keras

app = Flask(__name__)

#parameters = np.load('numpy_network/parameters.npy', allow_pickle=True)[()]

#Y_prediction_valid = np_pred(parameters, "numpy_network/camisa.jpg")

#model = keras.models.load_model("tf_network/ann_tf_fashmnist.h5")
#tf_pred(model, "tf_network/camiseta.jpg")

@app.route("/")
def welcome():
    return render_template(
                            "index.html"
    )