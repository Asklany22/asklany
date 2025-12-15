import os
import json
import zipfile
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global variables for model and labels
model = None
labels = None

def unzip_model():
    """Unzip model_package.zip to model_artifacts/"""
    model_zip = Path("model_package.zip")
    model_artifacts_dir = Path("model_artifacts")
    
    if not model_artifacts_dir.exists() and model_zip.exists():
        print("Unzipping model_package.zip...")
        with zipfile.ZipFile(model_zip, 'r') as zip_ref:
            zip_ref.extractall(model_artifacts_dir)
        print("Model unzipped successfully")
    else:
        if model_artifacts_dir.exists():
            print("Model artifacts directory already exists")
        else:
            print("Warning: model_package.zip not found")

def load_model():
    """Load TensorFlow SavedModel"""
    global model
    model_path = Path("model_artifacts/kaggle/working/model_saved")
    
    if model_path.exists():
        print(f"Loading model from {model_path}...")
        model = tf.saved_model.load(str(model_path))
        print("Model loaded successfully")
    else:
        print(f"Warning: Model not found at {model_path}")

def load_labels():
    """Load labels from JSON file"""
    global labels
    labels_path = Path("model_artifacts/labels.json")
    
    if labels_path.exists():
        print(f"Loading labels from {labels_path}...")
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        print("Labels loaded successfully")
    else:
        print(f"Warning: Labels file not found at {labels_path}")

def preprocess_image(image_file):
    """
    Preprocess image: resize to 256x256, convert to RGB, normalize to float32
    """
    # Open image with PIL
    image = Image.open(image_file)
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to 256x256
    image = image.resize((256, 256))
    
    # Convert to numpy array and normalize to float32
    image_array = np.array(image, dtype=np.float32)
    
    # Normalize to [0, 1] range
    image_array = image_array / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "labels_loaded": labels is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    Accepts form-data with 'image' field
    Returns: id, name, confidence
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    if labels is None:
        return jsonify({"error": "Labels not loaded"}), 500
    
    try:
        image_file = request.files['image']
        
        # Preprocess image
        processed_image = preprocess_image(image_file)
        
        # Run inference using the 'serve' signature
        if "serve" not in model.signatures:
            available_sigs = list(model.signatures.keys())
            return jsonify({
                "error": f"'serve' signature not found. Available signatures: {available_sigs}"
            }), 500
        
        infer = model.signatures["serve"]
        predictions = infer(tf.constant(processed_image))
        
        # Get the output tensor (might be in different keys)
        # Try common output keys
        if isinstance(predictions, dict):
            if len(predictions) == 0:
                return jsonify({"error": "Model returned empty predictions"}), 500
            # Try to find the output tensor
            output_key = list(predictions.keys())[0]
            output = predictions[output_key].numpy()
        else:
            output = predictions.numpy()
        
        # Apply softmax if not already applied
        output = output[0]  # Remove batch dimension
        if output.max() > 1.0 or output.min() < 0.0:
            # Apply softmax
            exp_output = np.exp(output - np.max(output))
            output = exp_output / np.sum(exp_output)
        
        # Get predicted class and confidence
        predicted_id = int(np.argmax(output))
        confidence = float(output[predicted_id])
        
        # Get label name
        if isinstance(labels, dict):
            if str(predicted_id) in labels:
                predicted_name = labels[str(predicted_id)]
            elif predicted_id in labels:
                predicted_name = labels[predicted_id]
            else:
                predicted_name = f"class_{predicted_id}"
        elif isinstance(labels, list):
            predicted_name = labels[predicted_id] if predicted_id < len(labels) else f"class_{predicted_id}"
        else:
            predicted_name = f"class_{predicted_id}"
        
        return jsonify({
            "id": predicted_id,
            "name": predicted_name,
            "confidence": confidence
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Initialize model and labels on startup
print("Initializing application...")
unzip_model()
load_model()
load_labels()
print("Application initialization complete")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
