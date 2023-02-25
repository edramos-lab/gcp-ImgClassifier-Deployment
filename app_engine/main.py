import tensorflow as tf
from flask import Flask, jsonify, request
import numpy as np
from PIL import Image
import wandb
from tensorflow.keras.models import load_model
import joblib 


app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to my Flask API to recognize baseplates</h1>"
        "</body>"
        "</html>"
    )
    return body
# Load the model
#model = tf.keras.models.load_model('model-best.h5')
#model = joblib.load("model-best.h5")
#wandb artifacts
run = wandb.init()
artifact = run.use_artifact('aimfg-california/Nigel-Baseplates-2022/model-rosy-meadow-247:v0', type='model')
artifact_dir = artifact.download()
logged_model_wandb =artifact_dir#'./artifacts/rosy-meadow-247/model-best.h5'
MODEL=load_model(logged_model_wandb)
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
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']

    # Open the image file using PIL
    image = Image.open(file)

    # Preprocess the image
    image = preprocess_image(image)

    # Make a prediction
    prediction = MODEL.predict(image)

    # Get the predicted class index
    predicted_class_index = np.argmax(prediction, axis=1)[0]

    # Get the predicted class name
    predicted_class_name = class_names[predicted_class_index]

    # Return the predicted class as a JSON object
    return jsonify({'class': predicted_class_name})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
