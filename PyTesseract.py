import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def image_to_text(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError("Unable to read image file.")
    except Exception as e:
        print("Error loading image:", e)
        return None
    
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print("Error converting image to grayscale:", e)
        return None
    
    try:
        extracted_text = pytesseract.image_to_string(gray_image)
        return extracted_text
    except Exception as e:
        print("Error extracting text using Tesseract OCR:", e)
        return None


image_path = r'C:\Users\jared\Projects\image_translation\image_translation\pytesseract-simple-python-optical-character-recognition-7.png'

extracted_text = image_to_text(image_path)

if extracted_text:
    print("Extracted Text:")
    print(extracted_text)
else:
    print("Failed to extract text from the image.")

