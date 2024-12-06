import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# === Paths to datasets and models ===
DATASET_PATH = "./archive5/kag2"
MODEL_PATH = "my_model_66.h5"
LAND_MODEL_PATH = 'land_classification_model_new.h5'  # Path to land classification model
LAND_DATASET_DIR = "dataset"

# === Load Deep Learning Models ===
crop_model = load_model(MODEL_PATH)
land_model = tf.keras.models.load_model(LAND_MODEL_PATH)

# === Land Classification Labels ===
land_class_labels = {
    0: 'agri',
    1: 'barrenland',
    2: 'urban',
    3: 'grassland'
}

# === Crop Classification Labels ===
def get_crop_class_labels():
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(224, 224),
        batch_size=64,
        class_mode="categorical",
        shuffle=False,
    )
    class_labels = generator.class_indices
    return {v: k for k, v in class_labels.items()}

crop_class_labels = get_crop_class_labels()

# === Helper Functions for Predictions ===
def predict_land(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.array(img_resized).reshape((1, 224, 224, 3)) / 255.0
    Y_pred = land_model.predict(img_array)
    predicted_index = np.argmax(Y_pred, axis=1)[0]
    predicted_class = land_class_labels[predicted_index]
    return img, predicted_class

def find_file_in_dataset(filename):
    possible_extensions = [".jpeg", ".jpg", ".png"]
    for subfolder in os.listdir(DATASET_PATH):
        subfolder_path = os.path.join(DATASET_PATH, subfolder)
        if os.path.isdir(subfolder_path):
            for ext in possible_extensions:
                full_path = os.path.join(subfolder_path, filename + ext)
                if os.path.exists(full_path):
                    return full_path
    return None

def predict_crop(filepath):
    img = cv2.imread(filepath)
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.array(img_resized).reshape((1, 224, 224, 3)) / 255.0
    predictions = crop_model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    return predicted_index, predictions[0][predicted_index]

def generate_prediction_plot(filepath, predicted_label, ground_truth_label):
    img = cv2.imread(filepath)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title(f"Ground Truth: {ground_truth_label}")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(img_rgb)
    plt.title(f"Predicted: {predicted_label}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("static/prediction_result.png")
    plt.close()

# === Routes for Land Classification ===
@app.route('/predict_land', methods=['POST'])
def predict_land_route():
    data = request.get_json()
    filename = data.get('filename')
    if not filename:
        return jsonify({"error": "Please provide the 'filename'."}), 400

    image_path = None
    for folder in land_class_labels.values():
        possible_image_path = os.path.join(LAND_DATASET_DIR, folder, filename + '.png')
        if os.path.exists(possible_image_path):
            image_path = possible_image_path
            actual_class = folder
            break

    if image_path is None:
        return jsonify({"error": f"Image '{filename}' not found in any folder."}), 400

    img, predicted_class = predict_land(image_path)
    if img is None:
        return jsonify({"error": f"Unable to load image from path: {image_path}"}), 400

    response = {
        "predicted_class": predicted_class,
        "ground_truth": actual_class
    }

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'Ground Truth: {actual_class}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'Predicted: {predicted_class}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("static2/Land_prediction_result.png")
    return jsonify(response)

@app.route('/static2/<filename>')
def serve_static(filename):
    return send_from_directory('static2', filename)

# === Routes for Crop Classification ===
@app.route("/predict_crop", methods=["POST"])
def predict_crop_route():
    data = request.get_json()
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "No filename provided"}), 400
    filepath = find_file_in_dataset(filename)
    if not filepath:
        return jsonify({"error": f"File {filename} not found in dataset"}), 404
    predicted_index, confidence = predict_crop(filepath)
    predicted_label = crop_class_labels[predicted_index]
    ground_truth_label = os.path.basename(os.path.dirname(filepath))
    generate_prediction_plot(filepath, predicted_label, ground_truth_label)
    return jsonify({
        "ground_truth": ground_truth_label,
        "predicted_label": predicted_label,
        "confidence": float(confidence),
    })

# === Run the Flask App ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)