from flask import Flask, render_template, request, redirect, url_for, flash, abort
import flask
from werkzeug.utils import secure_filename
import os
import sys
from numpy_network.np_predict import predict as np_pred
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

@app.route('/predict', methods=['GET','POST'])
def upload_file():
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
            return redirect(url_for('uploaded_file',filename=filename,prediction=prediction))
    else:
        return render_template("predict.html")

@app.route("/predicted")
def uploaded_file():
    filename = request.args.get('filename',None)
    prediction = request.args.get('prediction',None)
    try:
        return render_template('image.html', filepath='uploads/' + filename, filename=filename,prediction=prediction)
    except IndexError:
        abort(404)

#model = keras.models.load_model("tf_network/ann_tf_fashmnist.h5")
#tf_pred(model, "tf_network/camiseta.jpg")

@app.route("/")
def welcome():
    return render_template(
                            "index.html"
    )