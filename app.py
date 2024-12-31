from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import base64

app = Flask(__name__)

# Load the pre-trained object detection model
model = tf.saved_model.load(r'C:\Users\diptiban\Desktop\flask_app\blindman\saved_models')


def process_image(image_data):
    # Decode base64 image
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Here we would run the image through the model and get the detected objects.
    # For simplicity, we are returning a dummy response.
    result = {"object": "car", "speed": 60}  # Dummy data for demonstration.
    return result


@app.route('/object-detection', methods=['POST'])
def object_detection():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    image_data = base64.b64encode(image_file.read()).decode('utf-8')
    result = process_image(image_data)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
