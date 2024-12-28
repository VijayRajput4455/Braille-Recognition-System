import os
import cv2
import uuid
import json
import numpy as np
from flask import Flask, request, jsonify, Response
from braille_detection import BrailleDetection
from braille_detection_utils import BrailleUtils

# Constants
MODEL_PATH = "weights/Braille_Yolov11.pt"  # Path to the trained Braille detection model
BRAILLE_MAP_PATH = "braille_map.json"      # Path to the Braille map JSON
DETECTED_IMAGE_DIR = "imagepath/"          # Directory to save annotated images

# Initialize the BrailleUtils class
braille_utils = BrailleUtils(braille_map_path=BRAILLE_MAP_PATH)

# Flask app initialization
app = Flask(__name__)

# Route: Home endpoint
@app.route("/", methods=["GET", "POST"])
def home():
    """
    Home endpoint for testing the API.
    Returns a simple "Hello World" message for GET requests.
    """
    if request.method == "GET":
        return jsonify({"message": "Hello, World!"})
    return jsonify({"message": "Unsupported request method"}), 405
    
@app.route("/brailledetection")
def braillelanguagedetection():
    image_url_path = request.args.get("image_url", None)
    if os.path.exists(image_url_path):
        image = cv2.imread(image_url_path)
    else:
        image =  braille_utils.urls_to_image(image_url_path)

    if isinstance(image, np.ndarray):
        # Initialize the Braille detector
        braille_detector = BrailleDetection(model_path=MODEL_PATH, confidence_threshold=0.2)
        annotated_image, detected_classes, braille_output = braille_detector.run(image)

        image_id = str(uuid.uuid4())
        resultimagepath =str("imagepath/detected_"+image_id+".png")
        cv2.imwrite("imagepath/detected_"+image_id+".png",annotated_image)

        response  = json.dumps({"Annotated_image": resultimagepath,  "Detected_Name": detected_classes,"Braille_Detection": braille_output}, ensure_ascii=False)
        return Response(response, content_type="application/json; charset=utf-8")

    else:
        response  = {"Annotated_image": image_url_path,  "Detected_Name": "None","Braille_Detection": "None", "Error": "Invalid image URL"}
        return jsonify(response)
    
if __name__ == "__main__": 
    app.run(port = 5000, debug = True)