import re
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import easyocr
import pandas as pd

# Define regex pattern for voltage
def extract_voltage(text):
    patterns = [
        r'(\d+(\.\d+)?)\s*(V|volt|volts|Volt|Volts)',  # Matches simple volt patterns
        r'(\[\d+(\.\d+)?,\s*\d+(\.\d+)?\])\s*(V|volt|volts|Volt|Volts)',  # Matches ranges
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    
    return None

# Function to process image
def process_image(image_url):
    try:
        # Load image
        response = requests.get(image_url, timeout=10)  # Add timeout for image request
        if response.status_code == 200:  # Check if the response is successful
            try:
                image = Image.open(BytesIO(response.content))
            except UnidentifiedImageError:
                print(f"Unable to identify image from URL: {image_url}")
                return None
        else:
            print(f"Failed to retrieve image from URL: {image_url}")
            return None
        
        # Initialize OCR reader with GPU
        reader = easyocr.Reader(['en'], gpu=True)
        
        # Perform OCR
        result = reader.readtext(image)
        
        # Combine OCR results into a single string
        text = ' '.join([res[1] for res in result])
        
        # Extract voltage
        voltage = extract_voltage(text)
        
        return voltage

    except requests.RequestException as e:
        print(f"Error fetching image from URL: {image_url} - {e}")
        return None

# Load CSV
df = pd.read_csv('voltage.csv')

# Process each image and extract voltage
df['extracted_voltage'] = df['image_link'].apply(process_image)

# Save updated CSV
df.to_csv('updated_file.csv', index=False)

