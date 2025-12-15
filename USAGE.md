# API Usage Examples

## Health Check

Check if the API is running and if the model is loaded:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "labels_loaded": true
}
```

## Image Prediction

Send an image for classification:

```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@/path/to/your/image.jpg"
```

Expected response:
```json
{
  "id": 0,
  "name": "class_name",
  "confidence": 0.95
}
```

## Testing with Python

```python
import requests

# Health check
response = requests.get('http://localhost:8000/health')
print(response.json())

# Image prediction
with open('image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:8000/predict', files=files)
    print(response.json())
```

## Environment Variables

- `PORT`: Port number for the server (default: 8000)

## Running with Gunicorn

```bash
# Using environment variable
PORT=8000 gunicorn app:app --bind 0.0.0.0:$PORT

# Default port
gunicorn app:app --bind 0.0.0.0:8000
```

## Model Setup

The application expects:
1. `model_package.zip` in the root directory containing:
   - Model at path: `kaggle/working/model_saved/`
2. Labels file at: `model_artifacts/labels.json`

The labels JSON should be in one of these formats:

**Format 1: Dictionary with string keys**
```json
{
  "0": "class_name_0",
  "1": "class_name_1",
  "2": "class_name_2"
}
```

**Format 2: Dictionary with integer keys (in Python/code, not pure JSON)**
```python
{
  0: "class_name_0",
  1: "class_name_1",
  2: "class_name_2"
}
```

**Format 3: List**
```json
[
  "class_name_0",
  "class_name_1",
  "class_name_2"
]
```
