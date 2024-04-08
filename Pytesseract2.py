import cv2
import pytesseract
import requests
import numpy as np
from io import BytesIO

# Set path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to enhance text
    _, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return threshold_image

def image_to_text(image):
    try:
        # Preprocess the image
        preprocessed_image = preprocess_image(image)
    except Exception as e:
        print("Error preprocessing image:", e)
        return None
    
    try:
        # Extract text using Tesseract OCR
        extracted_text = pytesseract.image_to_string(preprocessed_image)
        return extracted_text
    except Exception as e:
        print("Error extracting text using Tesseract OCR:", e)
        return None

def read_image_from_url(image_url):
    try:
        # Download image from URL
        response = requests.get(image_url)
        response.raise_for_status()
        # Read image from response content
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print("Error reading image from URL:", e)
        return None

# Prompt the user to input the image URL
image_url = input("Enter the URL of the image: ")

# Read image from URL
image = read_image_from_url(image_url)

if image is not None:
    # Extract text from the image
    extracted_text = image_to_text(image)
    
    if extracted_text:
        print("Extracted Text:")
        print(extracted_text)
    else:
        print("Failed to extract text from the image.")
else:
    print("Failed to read image from URL.")

