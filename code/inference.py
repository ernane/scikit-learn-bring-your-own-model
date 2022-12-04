import os

import joblib

"""
    https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/using_sklearn.html

    input_fn: Takes request data and deserializes the data into an object for prediction.
    predict_fn: Takes the deserialized request object and performs inference against the loaded model.
    output_fn: Takes the result of prediction and serializes this according to the response content type.
"""


def predict_fn(input_object, model):
    """
    """
    print("calling model")
    predictions = model.predict(input_object)
    return predictions


def model_fn(model_dir):
    """
    """
    print("loading model.joblib from: {}".format(model_dir))
    loaded_model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return loaded_model
