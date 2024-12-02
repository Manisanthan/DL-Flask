import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from flask import Flask, request, jsonify
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# === Paths to datasets and models ===
DATASET_PATH = "./archive5/kag2"
MODEL_PATH = "my_model_66.h5"

# === Load Deep Learning Model ===
model = load_model(MODEL_PATH)

# === Data Preprocessing for Deep Learning ===
# Generate class labels for crop classification
def get_class_labels():
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

class_labels = get_class_labels()

# === Image-based Crop Classification Helper Functions ===
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
    predictions = model.predict(img_array)
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



@app.route('/')
def home():
    return "Flask API is running!"



# === Routes for Deep Learning Model ===
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "No filename provided"}), 400
    filepath = find_file_in_dataset(filename)
    if not filepath:
        return jsonify({"error": f"File {filename} not found in dataset"}), 404
    predicted_index, confidence = predict_crop(filepath)
    predicted_label = class_labels[predicted_index]
    ground_truth_label = os.path.basename(os.path.dirname(filepath))
    generate_prediction_plot(filepath, predicted_label, ground_truth_label)
    return jsonify({
        "ground_truth": ground_truth_label,
        "predicted_label": predicted_label,
        "confidence": float(confidence),
    })

# === Run the Flask App ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)