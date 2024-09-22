import cv2
import pytesseract
from PIL import Image
import requests
from io import BytesIO

# Path to your Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Image URL
image_url = "https://m.media-amazon.com/images/I/612mrlqiI4L.jpg"

# Fetch the image from the URL
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Function to perform OCR on the image and print the results
def ocr_image(image, rotation_angle):
    rotated_image = image.rotate(rotation_angle, expand=True)
    text = pytesseract.image_to_string(rotated_image, timeout=10)
    print(f"\n......TEXT FROM ANGLE.......{rotation_angle}....degree.....rotation:\n{text}\n")

# Perform OCR on the original image and rotated images
print("Original Image OCR:")
ocr_image(image, 0)

print("90 Degree Rotation OCR:")
ocr_image(image, 90)

print("180 Degree Rotation OCR:")
ocr_image(image, 180)

print("270 Degree Rotation OCR:")
ocr_image(image, 270)
