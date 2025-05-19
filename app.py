import os

# Disable GPU usage for environments like Railway (no GPU support)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the model once at startup
try:
    model = load_model(os.path.join(os.getcwd(), "model", "age_gender_model.keras"))
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error("Failed to load model: %s", e)
    model = None

@app.route('/', methods=['GET'])
def health_check():
    return "OK", 200

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        image_file = request.files['image']
        image = Image.open(image_file).convert('L')  # Grayscale
        image = image.resize((128, 128))
        img_array = np.array(image).reshape(1, 128, 128, 1) / 255.0

        pred_gender, pred_age = model.predict(img_array)
        gender = "Female" if pred_gender[0][0] > 0.5 else "Male"
        age = int(pred_age[0][0])

        return jsonify({
            'predicted_gender': gender,
            'predicted_age': age
        })

    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500
