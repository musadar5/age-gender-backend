import os
# Disable all GPU usage (important on Railway)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load the model
model = load_model("model/age_gender_model.keras")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image = Image.open(image_file).convert('L')  # Convert to grayscale
    image = image.resize((128, 128))
    img_array = np.array(image).reshape(1, 128, 128, 1) / 255.0

    pred_gender, pred_age = model.predict(img_array)
    gender = "Female" if pred_gender[0][0] > 0.5 else "Male"
    age = int(pred_age[0][0])

    return jsonify({
        'predicted_gender': gender,
        'predicted_age': age
    })


if __name__ == '__main__':
    # Use Railway-friendly port config
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
