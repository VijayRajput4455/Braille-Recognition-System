import streamlit as st
import time
import requests
from PIL import Image
import urllib
import cv2
from io import BytesIO
import numpy as np
from ultralytics import YOLO
from texttospeech import TextToSpeech
from braille_detection_utils import BrailleUtils
from braille_detection import BrailleDetection  # Assuming your class is in braille_detection.py
from paddle_ocr_reader import extract_filtered_alpha_words


# Constants
MODEL_PATH = "weights/Braille_Yolov11.pt"
BRAILLE_MAP_PATH = "braille_map.json"
TTS_VOICE_ID = 'com.apple.speech.synthesis.voice.Alex'
TTS_RATE = 150
TTS_VOLUME = 1.0
DEFAULT_IMAGE_PATH = r"assets/alpha.jpeg"  # Default image path

# Initialize dependencies
braille_utils = BrailleUtils(BRAILLE_MAP_PATH)
braille_detector = BrailleDetection(model_path=MODEL_PATH, confidence_threshold=0.5)
tts = TextToSpeech(rate=TTS_RATE, volume=TTS_VOLUME, voice_id=TTS_VOICE_ID)

# Apply custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f5f7fa;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f2f3f5;
        color: #34495e;
    }
    [data-testid="stSidebar"] h1, h2, h3, h4, h5, h6, label {
        color:  #34495e; /* Sidebar headings are now white */
    
    }
    
    /* Title styling */
    h1 {
        color: #34495e;
        text-align: center;
    }
    

    /* Button animation */
    div.stButton button {
        background-color: #e74c3c;
        border: none;
        border-radius: 8px;
        color: white;
        font-size: 1rem;
        padding: 10px 20px;
        transition: transform 0.3s ease, background-color 0.3s ease;
    }
    div.stButton button:hover {
        transform: scale(1.1);
        background-color: #c0392b;
    }
    
      /* Styling for the download button specifically */
    .stDownloadButton button {
        background-color: #e74c3c;
        border: none;
        border-radius: 8px;
        color: white;
        font-size: 1rem;
        padding: 10px 20px;
        transition: transform 0.3s ease, background-color 0.3s ease;
    }

    .stDownloadButton button:hover {
        transform: scale(1.1);
        background-color: #c0392b;
    }


    /* Animation for spinner */
    .spinner {
        border: 6px solid #ecf0f1;
        border-top: 6px solid #e74c3c;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: auto;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Custom loading animation */
    .loading-text {
        font-size: 1.5rem;
        font-weight: bold;
        color: #e74c3c;
        animation: fade 1.5s ease-in-out infinite;
        text-align: center;
        margin-top: 20px;
    }
    @keyframes fade {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    
    </style>
    """,
    unsafe_allow_html=True
)

# Application Title
st.title("🟢 Braille Detection System")

# Sidebar Configuration
st.sidebar.header("🔧 Configuration")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05
)
braille_detector.confidence_threshold = confidence_threshold

# Upload Braille Image
uploaded_image = st.sidebar.file_uploader(
    "📂 Upload an Image of Braille Text", 
    type=["jpg", "png", "jpeg"], 
    label_visibility="visible", 
    help="Drag and drop or click Browse to upload an image."
)

url_input = st.text_area("Enter Image URL", help="Paste an image URL and press Enter to load the image." , height=68)
load_image_button = st.sidebar.button("🔄 Load Image from URL")

# url_input = urls_to_image(url_input)

# Default image path
input_image = None

# Handle URL input or image upload
if uploaded_image:
    image_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    input_image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
elif load_image_button and url_input:
    try:
        input_image = braille_utils.urls_to_image(url_input)
    except Exception as e:
        st.error(f"Error fetching the image from URL: {e}")
else:
    # Load default image if neither image upload nor URL is provided
    input_image = cv2.imread(DEFAULT_IMAGE_PATH)

# Main Logic
if input_image is not None:
    try:
        animation_placeholder = st.empty()
        animation_placeholder.markdown('<div class="loading-text">🔍 Processing Image...</div>', unsafe_allow_html=True)
        # Run Braille detection
        start_time = time.time()
        with st.spinner("✨ Detecting Braille Patterns..."):
            annotated_image, detected_classes, braille_output = braille_detector.run(input_image)
            end_time = time.time()
            prediction_time = end_time - start_time
            animation_placeholder.empty()

        # Display results
        if annotated_image is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📤 Uploaded Image")
                st.image(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), use_container_width=True)

            with col2:
                st.subheader("📸 Detected Image")
                st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.success(f"✅ Detection completed successfully Inference time: {prediction_time:.2f}")

            # Display detected Braille patterns
            st.subheader("🔡 Detected Braille Patterns")
            st.text_area("Braille Output", braille_output, height=68)

            # Extract and display filtered alphabets
            extracted_text = extract_filtered_alpha_words(input_image)
            st.subheader("🔡 Extracted Alphabets")
            st.text_area("Text Output", extracted_text, height=68)

            # Create vertical layout for buttons
            st.subheader("🎛️ Actions")
            with st.container():
                # Create two columns
                col1, col2, col3 = st.columns(3)

                # Place the "Read Detected Classes" button in the first column
                with col1:
                    if st.button("🔊 Speak Detected Classes"):
                        tts.speak(detected_classes)

                # Place the "Download Detected Image" button in the second column
                with col2:
                    # Allow users to download the detected image
                    is_success, image_buffer = cv2.imencode(".jpg", annotated_image)
                    if is_success:
                        st.download_button(
                            label="💾 Download Detected Image",
                            data=image_buffer.tobytes(),
                            file_name="detected_image.jpg",
                            mime="image/jpeg",
                        )
                with col3:
                    # Allow users to download the detected image
                    is_success, image_buffer = cv2.imencode(".jpg", input_image)
                    if is_success:
                        st.download_button(
                            label="💾 Download Raw Image",
                            data=image_buffer.tobytes(),
                            file_name="raw_image.jpg",
                            mime="image/jpeg",
                        )
        else:
            st.error("❌ Failed to process the image. Please try again with a different image.")

    except Exception as error:
        st.error(f"⚠️ An error occurred: {error}")



# Add footer
st.markdown("""
    <style>
    footer {visibility: hidden;}
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: black;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
    <p>🟢 Created By Vijay</p>
    </div>
""", unsafe_allow_html=True)
