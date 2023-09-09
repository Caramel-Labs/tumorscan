from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# load model
model = load_model("./model/model.h5")


# resize images before inference
def preprocess_image(image):
    image = image.resize((256, 256))
    return image


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # get image data from request
        file = request.files["file"]

        # check whether file is an image
        if file:
            image = Image.open(file)
            processed_image = preprocess_image(image)

            # perform inference on image using model
            predictions = model.predict(processed_image)

            # return top class (if model outputs class probabilities)
            top_class_index = np.argmax(predictions)

            # customizable response format
            response = {
                "class_id": top_class_index,
                "class_label": "Class Label",  # Replace with your class labels
                "confidence": float(predictions[0][top_class_index]),
            }

            return jsonify(response)
        else:
            return jsonify({"error": "No file uploaded"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
