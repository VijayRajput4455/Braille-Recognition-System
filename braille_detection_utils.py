import numpy as np
import urllib
import cv2
import json
from utils.augmentations import letterbox

class BrailleUtils:
    """
    A utility class for handling Braille character conversion, YOLO detection box processing,
    and image preprocessing.
    """

    def __init__(self, braille_map_path="braille_map.json"):
        """
        Initialize the utility class.

        Args:
            braille_map_path (str): Path to the JSON file containing the Braille mapping.
        """
        self.braille_map_path = braille_map_path
        self.braille_map = self._load_braille_map()

    def _load_braille_map(self):
        """
        Load the Braille mapping from the JSON file.

        Returns:
            dict: A dictionary containing the Braille mapping.
        """
        try:
            with open(self.braille_map_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Braille map file not found at {self.braille_map_path}")

    def convert_to_braille_unicode(self, str_input):
        """
        Convert a string input to its Braille Unicode representation.

        Args:
            str_input (str): The input string to convert.

        Returns:
            str: The Braille Unicode representation.
        """
        return self.braille_map.get(str_input, "")

    def parse_xywh_and_class(self, boxes):
        """
        Process YOLO detection boxes and group them by line based on y-coordinate.

        Args:
            boxes (torch.Tensor): A tensor containing the detection boxes.

        Returns:
            list: A list of grouped boxes sorted by y-coordinate and then x-coordinate.
        """
        # Convert YOLO boxes to NumPy array
        new_boxes = np.zeros(boxes.shape)
        new_boxes[:, :4] = boxes.xywh.cpu().numpy()  # First 4 channels are xywh
        new_boxes[:, 4] = boxes.conf.cpu().numpy()  # 5th channel is confidence
        new_boxes[:, 5] = boxes.cls.cpu().numpy()  # 6th channel is class index

        # Sort boxes by y-coordinate
        new_boxes = new_boxes[new_boxes[:, 1].argsort()]

        # Compute threshold to break lines
        y_threshold = np.mean(new_boxes[:, 3]) // 2
        boxes_diff = np.diff(new_boxes[:, 1])
        threshold_index = np.where(boxes_diff > y_threshold)[0]

        # Group boxes by line based on threshold index
        boxes_clustered = np.split(new_boxes, threshold_index + 1)
        boxes_return = []

        for cluster in boxes_clustered:
            # Sort each cluster by x-coordinate
            cluster = cluster[cluster[:, 0].argsort()]
            boxes_return.append(cluster)

        return boxes_return

    @staticmethod
    def load_image(image, new_shape=640):
        """
        Preprocess an image for YOLO detection.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            new_shape (int): The target shape for resizing.

        Returns:
            tuple: Preprocessed image and the original image.
        """
        original_image = image.copy()
        image = letterbox(original_image, new_shape=new_shape, stride=32)[0]

        # Convert BGR to RGB and adjust dimensions for YOLO input
        image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        image = np.ascontiguousarray(image)
        return image, original_image

    @staticmethod
    def urls_to_image(image_url):
        """
        Load an image from a URL or a local file path.

        Args:
            image_url (str): The image URL or local path.

        Returns:
            np.ndarray: Loaded image as a NumPy array.
        """
        try:
            if image_url.startswith("http"):
                req = urllib.request.urlopen(image_url)
                arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                image = cv2.imdecode(arr, -1)
                return image
            else:
                image = cv2.imread(image_url)
                return image
        except Exception as e:
            print("Error occurred:", e)
            return None
