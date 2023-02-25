import joblib
import numpy as np 
from PIL import Image
from flask import Flask, jsonify, request
from google.cloud import storage

storage_client = storage.Client()
bucket = storage_client.get_bucket("model-baseplates")
blob = bucket.blob("model-best.h5")
blob.download_to_filename("/tmp/model-best.h5")
model = joblib.load("/tmp/model-best.h5")
# Define the class names
class_names = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6']

# Define a function to preprocess the image
def preprocess_image(image):
    image = image.resize((299, 299))
    image = np.array(image)
    image = image/255.0#tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# Define the prediction route

def predict():
    # Get the image file from the request
    file = request.files['image']

    # Open the image file using PIL
    image = Image.open(file)

    # Preprocess the image
    image = preprocess_image(image)

    # Make a prediction
    prediction = model.predict(image)

    # Get the predicted class index
    predicted_class_index = np.argmax(prediction, axis=1)[0]

    # Get the predicted class name
    predicted_class_name = class_names[predicted_class_index]

    # Return the predicted class as a JSON object
    return jsonify({'class': predicted_class_name})
