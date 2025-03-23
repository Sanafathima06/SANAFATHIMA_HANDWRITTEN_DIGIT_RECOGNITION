import tensorflowjs as tfjs
from tensorflow.keras.models import load_model
import os

# Load trained model
model = load_model("model/mnist_cnn.h5")

# Create webapp/model directory if not exists
if not os.path.exists("webapp/model"):
    os.makedirs("webapp/model")

# Convert to TensorFlow.js format
tfjs.converters.save_keras_model(model, "webapp/model")
print("Model converted and saved successfully!")
