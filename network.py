from flask import Flask, render_template, request, redirect, url_for, flash, abort
from werkzeug.utils import secure_filename
import os
from numpy_network.np_predict import predict as np_pred
from tf_network.tf_predict import predict as tf_pred
from numpy_network.evaluate import evaluate
from numpy_network.data_prep import data_preparation
import numpy as np
from tensorflow import keras

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def welcome():
    return render_template(
                            "index.html"
    )

@app.route("/info")
def info():
    return render_template(
                            "info.html"
    )

@app.route('/np_predict', methods=['GET','POST'])
def np_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            parameters = np.load('numpy_network/parameters.npy', allow_pickle=True)[()]
            fpath = 'static/uploads/'+ filename
            prediction = np_pred(parameters,fpath)[0]
            return redirect(url_for('uploaded_file',filename=filename,prediction=prediction,network="Numpy"), code=307)
    else:
        return render_template("predict.html", network="Numpy")

@app.route('/tf_predict', methods=['GET','POST'])
def tf_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            model = keras.models.load_model("tf_network/ann_tf_fashmnist.h5")
            fpath = 'static/uploads/' + filename
            prediction = tf_pred(model,fpath)
            return redirect(url_for('uploaded_file',filename=filename,prediction=prediction,network="Tensorflow"), code=307)
    else:
        return render_template("predict.html", network="Tensorflow")

@app.route("/evaluate_np")
def evaluate_np():
    """
    evaluate the model
    """
    X_train, y_train, X_valid, y_valid, X_test, y_test = data_preparation()

    parameters = np.load('numpy_network/parameters.npy', allow_pickle=True)[()]

    Y_prediction_train = evaluate(parameters,X_train,y_train)
    y_train = np.argmax(y_train, axis=0)
    train_accu = np.mean(Y_prediction_train == y_train) * 100

    Y_prediction_valid = evaluate(parameters,X_valid,y_valid)
    y_valid = np.argmax(y_valid, axis=0)
    validation_accu = np.mean(Y_prediction_valid == y_valid) * 100

    Y_prediction_test = evaluate(parameters,X_test,y_test)
    y_test = np.argmax(y_test, axis=0)
    test_accu = np.mean(Y_prediction_test == y_test) * 100
    return render_template('stats.html', train_accu=train_accu, test_accu=test_accu, validation_accu=validation_accu)

@app.route("/predicted", methods=['POST'])
def uploaded_file():
    
    if request.method == 'POST':
        filename = request.args.get('filename',None)
        prediction = request.args.get('prediction',None)
        network = request.args.get('network',None)
        try:
            return render_template('image.html', filepath='uploads/' + filename, filename=filename,prediction=prediction,network=network)
        except IndexError:
            abort(404)
    else:
        abort(403)
