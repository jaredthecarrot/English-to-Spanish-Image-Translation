import cv2
import pytesseract

# Set path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert RGB to grayscale
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

# Open the default camera (usually the first one)
camera = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    
    if not ret:
        print("Failed to capture frame")
        break
    
    # Display the captured frame
    cv2.imshow('Camera', frame)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Convert frame from RGB to BGR
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Extract text from the captured frame
    extracted_text = image_to_text(frame_bgr)
    
    if extracted_text:
        print("Extracted Text:")
        print(extracted_text)
    else:
        print("Failed to extract text from the image.")

# Release the camera and close OpenCV windows
camera.release()
cv2.destroyAllWindows()