import os
from PIL import Image
import numpy as np
from ultralytics import YOLO
from convert import convert_to_braille_unicode, parse_xywh_and_class
from texttospeech import TextToSpeech

class BrailleDetection:
    """
    A class for detecting Braille characters in images using YOLO and converting the results to Braille Unicode.
    """

    def __init__(self, model_path, confidence_threshold=0.5):
        """
        Initialize the BrailleDetection class.

        Args:
            model_path (str): Path to the YOLO model weights file.
            confidence_threshold (float): Confidence threshold for detection.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = self._load_model()

    def _load_model(self):
        """
        Load the YOLO model from the specified path.

        Returns:
            YOLO: The loaded YOLO model.
        """
        return YOLO(self.model_path)

    def _predict(self, image_path):
        """
        Run predictions on the given image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            tuple: (YOLO Results, Numpy Image of predicted image) or (None, None) in case of error.
        """

        # Run predictions
        try:
            results = self.model.predict(
                source=image_path,
                save=False,  # Disable saving to control results manually
                conf=self.confidence_threshold
            )
            # Convert prediction result to a PIL image with bounding boxes
            annotated_image = results[0].plot()  # This returns a NumPy array
            return results, annotated_image
        except Exception as e:
            print(f"Error: Prediction failed for image '{image_path}'. Exception: {e}")
            return None, None

    def _process_predictions(self, results):
        """
        Process the model predictions and convert detected classes to Braille Unicode.

        Args:
            results (YOLO Results): YOLO prediction results.

        Returns:
            tuple: Detected classes as a string and their Braille Unicode representation.
        """
        if results is None:
            return "", ""

        boxes = results[0].boxes  # Assuming predictions for a single image
        list_boxes = parse_xywh_and_class(boxes)

        detected_classes = ""
        braille_representation = ""

        for box_line in list_boxes:
            braille_line = ""
            box_classes = box_line[:, -1]  # Get class indices
            for each_class in box_classes:
                class_name = self.model.names[int(each_class)]  # Retrieve class name
                detected_classes += class_name
                braille_line += convert_to_braille_unicode(class_name)  # Convert class name to Braille Unicode

            braille_representation += braille_line + "\n"

        return detected_classes, braille_representation

    def run(self, image_path):
        """
        Run the full detection pipeline on the given image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            tuple: Annotated image, detected classes, and Braille output, or (None, None, None) in case of error.
        """
        results, annotated_image = self._predict(image_path)
        if annotated_image is None:
            return None, None, None
        
        detected_classes, braille_output = self._process_predictions(results)
        return annotated_image, detected_classes, braille_output


# Example usage
if __name__ == "__main__":
    # Path to the YOLO model weights
    MODEL_PATH = "weights/Braille_Yolov11.pt"
    
    # Path to the image to process (test with an incorrect path to trigger the error handling)
    IMAGE_PATH = r"C:\Users\rajpu\Desktop\VIJAY\Projects\hhh\DataSet\000001.jpg"

    # Initialize the BrailleDetection class
    braille_detector = BrailleDetection(model_path=MODEL_PATH, confidence_threshold=0.5)
    tts = TextToSpeech(rate=150, volume=1.0, voice_id='com.apple.speech.synthesis.voice.Alex')

    # Run the detection pipeline and get the results
    annotated_image, detected_classes, braille_output = braille_detector.run(IMAGE_PATH)

    if annotated_image is not None:
        # Display the annotated image (for debugging/visualization)
        # annotated_image.show()

        # Print the output
        print(f"Detected Classes: {detected_classes}")
        tts.speak(detected_classes)
    else:
        print(f"Failed to process the image at '{IMAGE_PATH}'. Please check the file path and try again.")
