from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your exported model
model = tf.keras.models.load_model("best_model.keras")

# Define class names (same as training)
class_names = ["Abnormal(Ulcer)", "Normal(Healthy skin)"]

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # Get uploaded file
    file = request.files["file"]
    img = Image.open(file).resize((224, 224))  # resize to model input
    arr = np.expand_dims(np.array(img)/255.0, axis=0)  # normalize
  
    # Run prediction
    pred = model.predict(arr)
    class_index = int(np.argmax(pred))
    confidence = float(np.max(pred))
    
    return jsonify({
        "prediction": class_names[class_index],
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
