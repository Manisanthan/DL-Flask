import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, send_from_directory

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# === Paths to models ===
LAND_MODEL_PATH = "land_classification_model_new.h5"  # Land classification model path
CROP_MODEL_PATH = "my_model_66.h5"  # Crop classification model path

# === Load Deep Learning Models ===
try:
    land_model = load_model(LAND_MODEL_PATH)
    print("Land model loaded successfully!")
except Exception as e:
    print(f"Error loading land model: {e}")
    land_model = None

try:
    crop_model = load_model(CROP_MODEL_PATH)
    print("Crop model loaded successfully!")
except Exception as e:
    print(f"Error loading crop model: {e}")
    crop_model = None

# === Class Labels ===
# Replace these with your actual class labels from your dataset
land_class_labels = {0: 'agri',1: 'barrenland',2: 'urban',3: 'grassland'}
crop_class_labels = {0: "jute", 1: "maize", 2: "rice",3:"sugarcane",4:"wheat"}

# Preprocess the image directly from memory
def preprocess_image(file, target_size=(224, 224)):
    img = Image.open(file)
    img = img.resize(target_size)  # Resize image
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Generate prediction plot and save it to the specified static folder
def generate_prediction_plot(img, predicted_label, static_dir, result_filename):
    try:
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f"Predicted: {predicted_label}")
        plt.axis("off")

        # Save the plot to the specified static directory with the given filename
        os.makedirs(static_dir, exist_ok=True)
        result_path = os.path.join(static_dir, result_filename)
        plt.savefig(result_path)
        plt.close()
        return result_path
    except Exception as e:
        print(f"Error generating prediction plot: {e}")
        return None

# === API Routes ===
# Land classification prediction endpoint
@app.route('/static2/<path:filename>')
def serve_static2(filename):
    return send_from_directory('static2', filename)

@app.route('/predict_land', methods=['POST'])
def predict_land():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Preprocess the image directly from memory
        img_array = preprocess_image(file)
        img = Image.open(file)
        img_rgb = img.convert("RGB")

        # Predict the class using the land model
        predictions = land_model.predict(img_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        # Get the predicted label for land
        predicted_label = land_class_labels[predicted_index]

        # Generate and save prediction plot (replace existing image)
        generate_prediction_plot(img_rgb, predicted_label, static_dir="static2", result_filename="Land_prediction_result.png")

        return jsonify({
            'predicted_label': predicted_label,
            'confidence': float(confidence),
            'result_image_url': "/static2/Land_prediction_result.png"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Crop classification prediction endpoint
@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Preprocess the image directly from memory
        img_array = preprocess_image(file)
        img = Image.open(file)
        img_rgb = img.convert("RGB")

        # Predict the class using the crop model
        predictions = crop_model.predict(img_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        # Get the predicted label for crops
        predicted_label = crop_class_labels[predicted_index]

        # Generate and save prediction plot (replace existing image)
        generate_prediction_plot(img_rgb, predicted_label, static_dir="static", result_filename="prediction_result.png")

        return jsonify({
            'predicted_label': predicted_label,
            'confidence': float(confidence),
            'result_image_url': "/static/prediction_result.png"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === Run the Flask App ===
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
