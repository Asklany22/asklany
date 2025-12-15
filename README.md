# asklany

Flask API for image classification using TensorFlow SavedModel.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure `model_package.zip` is in the root directory. The application will automatically unzip it to `model_artifacts/` on startup.

3. Ensure labels JSON file is at `model_artifacts/labels.json`.

## Running the Application

### With Gunicorn (Production)
```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

### With Flask Development Server
```bash
python app.py
```

## API Endpoints

### GET /health
Health check endpoint that returns the status of the application and whether the model and labels are loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "labels_loaded": true
}
```

### POST /predict
Prediction endpoint that accepts an image file and returns the predicted class.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `image` field with an image file

**Response:**
```json
{
  "id": 0,
  "name": "class_name",
  "confidence": 0.95
}
```

## Image Processing

Images are automatically:
- Resized to 256x256 pixels
- Converted to RGB format
- Normalized to float32 [0, 1] range

## CORS

CORS is enabled for all origins.