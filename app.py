from flask import Flask, render_template_string, request, jsonify
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2
import numpy as np
import io
import os
from PIL import Image
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

# Constants
IMG_SIZE = 224
CHANNELS = 3
N_CLASSES = 4
CLASS_LABELS = ['Cataract', 'Diabetic_Retinopathy', 'Glaucoma', 'Normal']
MODEL_PATH = 'resnet50_eye_model.keras'

def create_model(weights_path=None):
    """
    Create and return the model with optional weight loading
    """
    logging.info("Creating model architecture...")
    
    # Create the base model with pretrained weights
    base_model = ResNet50(
        weights='imagenet',  # Load imagenet weights
        include_top=False,   # Don't include the classification layers
        input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS),
        pooling='max'
    )
    
    # Create the model
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS))
    x = base_model(inputs)
    
    # Add custom layers
    x = layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = layers.Dropout(rate=0.5, seed=123)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(N_CLASSES, activation='softmax', kernel_regularizer=l2(0.001))(x)
    
    # Create the full model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Load weights if provided
    if weights_path and os.path.exists(weights_path):
        logging.info(f"Loading weights from {weights_path}")
        model.load_weights(weights_path)
    
    return model

# Image validation function
def validate_image(image):
    if image.size[0] < 10 or image.size[1] < 10:
        raise ValueError("Image dimensions too small. Minimum size is 10x10 pixels.")
    if image.size[0] > 4096 or image.size[1] > 4096:
        raise ValueError("Image dimensions too large. Maximum size is 4096x4096 pixels.")
    if image.mode not in ['RGB', 'L']:
        raise ValueError("Unsupported image mode. Please use RGB or grayscale images.")

def preprocess_image(image):
    # Validate image
    validate_image(image)
    
    # Convert image to RGB if it's not
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image to match model's expected sizing
    image = image.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert image to numpy array and normalize
    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

# HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Eye Disease Classification</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f0f2f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
        h1 {
            color: #1a237e;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        .upload-section {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            border: 2px dashed #3f51b5;
            border-radius: 10px;
            background-color: #f5f5f5;
        }
        .preview-section {
            text-align: center;
            margin-bottom: 30px;
        }
        #imagePreview {
            max-width: 400px;
            max-height: 400px;
            margin: 20px auto;
            display: none;
            border: 3px solid #3f51b5;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .result-section {
            text-align: center;
            margin-top: 20px;
            padding: 20px;
            border-radius: 10px;
            display: none;
            background-color: #e8eaf6;
        }
        .submit-btn {
            background-color: #3f51b5;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .submit-btn:hover {
            background-color: #283593;
        }
        .submit-btn:disabled {
            background-color: #9fa8da;
            cursor: not-allowed;
        }
        .error-message {
            color: #d32f2f;
            background-color: #ffebee;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Eye Disease Classification</h1>
        
        <div class="upload-section">
            <input type="file" id="imageInput" accept="image/*" onchange="previewImage(this)">
        </div>
        
        <div class="preview-section">
            <img id="imagePreview">
        </div>
        
        <div class="error-message" id="errorMessage"></div>
        
        <div style="text-align: center;">
            <button class="submit-btn" onclick="classifyImage()" id="classifyBtn" disabled>
                Classify Image
            </button>
        </div>
        
        <div class="result-section" id="resultSection">
            <div id="prediction"></div>
            <div id="confidence"></div>
        </div>
    </div>

    <script>
        function showError(message) {
            const errorDiv = document.getElementById('errorMessage');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
            setTimeout(() => {
                errorDiv.style.display = 'none';
            }, 5000);
        }

        function previewImage(input) {
            const preview = document.getElementById('imagePreview');
            const file = input.files[0];
            const classifyBtn = document.getElementById('classifyBtn');
            
            if (file) {
                // Validate file type
                if (!file.type.startsWith('image/')) {
                    showError('Please upload an image file.');
                    input.value = '';
                    preview.style.display = 'none';
                    classifyBtn.disabled = true;
                    return;
                }
                
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                    classifyBtn.disabled = false;
                    document.getElementById('resultSection').style.display = 'none';
                }
                reader.readAsDataURL(file);
            } else {
                preview.style.display = 'none';
                classifyBtn.disabled = true;
            }
        }
        
        function classifyImage() {
            const input = document.getElementById('imageInput');
            if (!input.files[0]) {
                showError('Please select an image first');
                return;
            }
            
            const formData = new FormData();
            formData.append('image', input.files[0]);
            
            const classifyBtn = document.getElementById('classifyBtn');
            classifyBtn.disabled = true;
            
            fetch('/classify', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                classifyBtn.disabled = false;
                
                if (data.error) {
                    showError(data.error);
                    return;
                }
                
                const resultSection = document.getElementById('resultSection');
                resultSection.style.display = 'block';
                document.getElementById('prediction').textContent = 
                    `Prediction: ${data.prediction}`;
                document.getElementById('confidence').textContent = 
                    `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
            })
            .catch(error => {
                classifyBtn.disabled = false;
                showError('Error classifying image. Please try again.');
                console.error('Error:', error);
            });
        }
    </script>
</body>
</html>
'''

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read and validate image
        try:
            image = Image.open(io.BytesIO(file.read()))
            validate_image(image)
        except Exception as e:
            return jsonify({'error': f'Invalid image: {str(e)}'}), 400
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Get prediction
        predictions = model.predict(processed_image)
        
        # Get the predicted class and confidence
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])
        
        # Get the class label
        predicted_class = CLASS_LABELS[predicted_class_index]
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence
        })
        
    except Exception as e:
        logging.error(f"Error during classification: {str(e)}")
        return jsonify({
            'error': 'An error occurred during classification'
        }), 500

# Initialize the model
try:
    # Create model and load weights in one step
    model = create_model(weights_path=MODEL_PATH)
    
    # Validate model with a dummy prediction
    dummy_input = np.zeros((1, IMG_SIZE, IMG_SIZE, CHANNELS))
    _ = model.predict(dummy_input)
    logging.info("Model loaded and validated successfully!")
except Exception as e:
    logging.error(f"Error initializing model: {str(e)}")
    raise

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)