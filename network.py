from flask import Flask, render_template, request, redirect, url_for, flash, abort
from werkzeug.utils import secure_filename
import os
from numpy_network.np_predict import predict as np_pred
from tf_network.tf_predict import predict as tf_pred
from numpy_network.evaluate import evaluate, plot
from numpy_network.data_prep import data_preparation
import numpy as np
from tensorflow import keras
import gc
import json

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

@app.route('/train', methods=['GET','POST'])
def train_np():
    """
    testing function
    """
    if request.method == 'GET':
        return render_template("train.html")
    else:
        return redirect(url_for('evaluate_np'), code=307)

@app.route("/evaluate_np", methods=['GET','POST'])
def evaluate_np():
    """
    evaluate the model
    """
    if request.method == "POST":
        X_train, y_train, X_valid, y_valid, X_test, y_test = data_preparation()

        parameters = np.load('numpy_network/parameters.npy', allow_pickle=True)[()]

        Y_prediction_train, conf_mat_train, class_rep_train, class_names = evaluate(parameters,X_train,y_train,"train_scores")
        y_train = np.argmax(y_train, axis=0)
        train_accu = np.mean(Y_prediction_train == y_train) * 100

        Y_prediction_valid, conf_mat_valid, class_rep_valid, class_names = evaluate(parameters,X_valid,y_valid,"valid_scores")
        y_valid = np.argmax(y_valid, axis=0)
        validation_accu = np.mean(Y_prediction_valid == y_valid) * 100

        Y_prediction_test, conf_mat_test, class_rep_test, class_names = evaluate(parameters,X_test,y_test,"test_scores")
        y_test = np.argmax(y_test, axis=0)
        test_accu = np.mean(Y_prediction_test == y_test) * 100

        train_matrix = plot(conf_mat_train, class_names, "train_matrix")
        valid_matrix = plot(conf_mat_valid, class_names,  "valid_matrix")
        test_matrix = plot(conf_mat_test, class_names, "test_matrix")

        del Y_prediction_train, Y_prediction_test, Y_prediction_valid, parameters, X_train, y_train, X_valid, y_valid, X_test, y_test
        gc.collect()

        return render_template('stats.html', train_accu=train_accu,
                        test_accu=test_accu,
                        validation_accu=validation_accu,
                        conf_mat_test=conf_mat_test,
                        class_names=class_names,
                        train_url=train_matrix,
                        valid_url=valid_matrix,
                        class_rep_train=class_rep_train,
                        class_rep_valid=class_rep_valid,
                        class_rep_test=class_rep_test,
                        test_url=test_matrix)
    else:
        with open('numpy_network/train_scores.json', 'r') as openfile:
            train_scores = json.load(openfile)
        with open('numpy_network/valid_scores.json', 'r') as openfile:
            valid_scores = json.load(openfile)
        with open('numpy_network/test_scores.json', 'r') as openfile:
            test_scores = json.load(openfile)
        return render_template('stats.html',
                                            class_rep_train=train_scores,
                                            class_rep_valid=valid_scores,
                                            class_rep_test=test_scores)


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
