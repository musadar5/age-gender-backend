from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
from tensorflow.keras.models import load_model
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load model
model = load_model("model/age_gender_model.keras")

# Prediction function
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image = Image.open(image_file).convert('L')
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
    app.run(debug=True)

