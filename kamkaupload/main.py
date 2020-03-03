from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer
from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import sys
import os
import glob
import re

detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.73.hdf5'



face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry","disgust","fear","happy","sad","surprised","neutral"]


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

def model_predict(img_path):

    emotion_classifier = load_model(emotion_model_path, compile=False)
    frame = cv2.imread(img_path,1)
    frame = imutils.resize(frame,width=300)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    canvas = np.zeros((250, 300, 3), dtype='uint8')
    frameClone = frame.copy()

    faces = sorted(faces, reverse=True,
    key=lambda x: (x[2] - x[0])*(x[3]-x[1]))[0]
    (fX, fY, fW, fH) = faces

    roi = gray[fY:fY + fH, fX:fX + fW]

    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    preds = emotion_classifier.predict(roi)[0]

    return preds



@app.route("/")
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path)

        
        label = EMOTIONS[preds.argmax()]

        return label
    return None


app.run(debug=True)
