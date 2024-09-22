import re
import requests
from PIL import Image
from io import BytesIO
import easyocr

# Function to extract values with a number and a unit
def extract_number_unit_pairs(text):
    # Regex pattern to match numbers followed by units (including floating point numbers)
    pattern = r'(\d+(\.\d+)?)\s*([a-zA-Z]+)'
    matches = re.findall(pattern, text)
    return matches

# Load and process image
def process_image(image_url):
    # Load image
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    
    # Initialize OCR reader
    reader = easyocr.Reader(['en'])
    
    # Perform OCR
    result = reader.readtext(image)
    
    # Combine OCR results into a single string
    text = ' '.join([res[1] for res in result])
    print("original_text : ", text)
    
    # Extract number-unit pairs
    number_unit_pairs = extract_number_unit_pairs(text)
    
    return number_unit_pairs

# Example image URL
image_url = 'https://m.media-amazon.com/images/I/71+SYPZhxuL.jpg'
number_unit_pairs = process_image(image_url)

# Print extracted number-unit pairs
for number, _, unit in number_unit_pairs:
    print(f'Value: {number}, Unit: {unit}')
