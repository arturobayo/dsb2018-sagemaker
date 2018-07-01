# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import StringIO
import sys
import signal
import traceback

import flask

import pandas as pd
import numpy as np

import tensorflow as tf
from keras.models import load_model
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from unet import mean_iou

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# Set size standards for the images we're going to process
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

# Class NumPyArangeEncoder that allows to convert a ndarray into a json object
class NumPyArangeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist() # or map(int, obj)
        return json.JSONEncoder.default(self, obj)


# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            cls.model = load_model(os.path.join(model_path, 'unet.h5'), custom_objects={'mean_iou': mean_iou})
            global graph
            graph = tf.get_default_graph()
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input: The data on which to do the predictions."""
        clf = cls.get_model()

        with graph.as_default():
            prediction = clf.predict(input, verbose=1)
        return prediction

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this server, we take data as PNG, convert
    it to a Numpy Array for internal use and then convert the predictions back to JSON
    """
    data = None
    X = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)

    # Convert from CSV to pandas
    if flask.request.content_type == 'image/png':
        data = flask.request.data
        img = imread(StringIO.StringIO(data))[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X[0] = img
    else:
        return flask.Response(response='This predictor only supports PNG images as input', status=415, mimetype='text/plain')

    # Do the prediction
    predictions = ScoringService.predict(X)

    # Convert predictions to JSON and send the response
    return flask.Response(response=json.dumps(predictions, cls=NumPyArangeEncoder), status=200, mimetype='application/json')
