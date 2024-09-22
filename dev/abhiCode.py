import easyocr
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import re
import difflib

# Function to extract voltages from text
def extract_volts(text):
    # Updated regex to capture voltage ranges like '100-240V' as well as single voltages like '48V'
    pattern = r'(\d+(\.\d+)?)[\s]*-[\s]*(\d+(\.\d+)?)[\s]*[vV]\b|\b(\d+(\.\d+)?)[\s]*[vV](?:olts?)?\b|\b[vV](\d+(\.\d+)?)\b'
    
    # Search for matches in the text
    matches = re.findall(pattern, text)
    
    unique_volts = set()  # Use a set to store unique voltage values
    
    for match in matches:
        # Handle voltage ranges like '100-240V'
        if match[0] and match[2]:
            unique_volts.add(f"[{float(match[0])},{float(match[2])}] volt")
        # Handle single voltage values
        elif match[4]:
            unique_volts.add(f"{float(match[4])} volt")
        elif match[5]:
            unique_volts.add(f"{float(match[5])} volt")
    
    return list(unique_volts)  


def extract_volume(text):
    text = text.replace("IZ", "12").replace(",",".")
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = ' '.join(text.split())
    text = text.lower()
    
    # Enhanced regex to handle units with potential noise (e.g., 'f oz', '128 f 0z', etc.)
    pattern = r'(\d+(\.\d+)?)[\s]*(f[\s]*[o0]z|fl[\s]*[o0]z|[o0]z|m[lL]|[lL]|litre|litri|gallon|gal|gallons?)|\b(f[\s]*[o0]z|fl[\s]*[o0]z|[o0]z|m[lL]|[lL]|litre|litri|gallon|gal|gallons?)\s*(\d+(\.\d+)?)\b'

    # Allowed units for filtering
    allowed_units = {
        'centilitre': ['cl'],
        'cubic foot': ['cu ft', 'cubic foot'],
        'cubic inch': ['cu in', 'cubic inch'],
        'cup': ['cup', 'cups'],
        'decilitre': ['dl', 'decilitre'],
        'fluid ounce': ['fl oz', 'f oz', 'oz', 'fluid ounce'],
        'gallon': ['gallon', 'gallons', 'gal'],
        'imperial gallon': ['imperial gallon'],
        'litre': ['l', 'litre', 'litri', 'litres'],
        'microlitre': ['microlitre'],
        'millilitre': ['ml', 'millilitre', 'millilitres'],
        'pint': ['pint', 'pints'],
        'quart': ['quart', 'quarts']
    }

    # Function to map extracted unit to allowed units
    def map_to_allowed_unit(unit):
        unit = unit.lower()
        for allowed_unit, aliases in allowed_units.items():
            if any(alias in unit for alias in aliases):
                return allowed_unit
        return None
    
    # Search for matches in the text
    matches = re.findall(pattern, text)
    
    unique_volumes = set()  # Use a set to store unique volume values
    
    for match in matches:
        volume_value = match[0]  # First capturing group holds the volume value
        unit = match[2]  # Capture the unit of volume
        
        if volume_value and unit:
            volume_value_clean = re.sub(r'\s+', '', volume_value)  # Clean spaces within the volume value
            mapped_unit = map_to_allowed_unit(unit)
            if mapped_unit:
                unique_volumes.add(f"{float(volume_value_clean)} {mapped_unit}")
    
    return list(unique_volumes)

def process_image(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    
    reader = easyocr.Reader(['en'])
    
    best_text = ""
    max_length = 0
    
    for angle in [0, 90, 180, 270]:
        rotated_image = image.rotate(angle, expand=True)
        
        rotated_image_np = np.array(rotated_image)
        result = reader.readtext(rotated_image_np)
        
        text = ' '.join([res[1] for res in result])
        
        if len(text) > max_length:
            best_text = text
            max_length = len(text)
    
    print("best_text : ", best_text)
    
    return best_text


def extract_wattage(text, pattern):
    # print("OG TXT : ", text)
    # # Replace common OCR errors and fix known issues
    # text = text.replace("IZ", "12").replace(",", ".").replace("S0", "50")
    # # Correct misinterpreted 'oo' to '00'
    # text = text.replace("oo", "00")
    
    # # Fix any remaining 'O' that might be misinterpreted as '0'
    # text = text.replace("O", "0")
    # # Handle common misinterpretations and errors
    # text = text.replace("1O", "10").replace("O", "0")
    
    # # Add space between digits and letters, and between letters and digits
    # #text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    # #text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    
    # # Remove unwanted special characters while keeping alphanumeric characters, spaces, dots, and dashes
    # text = re.sub(r'[^a-zA-Z0-9\s.\-\[\]]', '', text)
    # text = text.replace('*', '-')
    # # Normalize whitespace
    # text = ' '.join(text.split())
    
    # # Convert to lowercase
    # text = text.lower()

    print("OG TXT : ", text)
    
    # Replace common OCR errors and fix known issues
    text = text.replace("IZ", "12").replace(",", ".").replace("S0", "50")
    text = text.replace("oo", "00")
    text = text.replace("O", "0")
    text = text.replace("1O", "10").replace("O", "0")
    
    # Replace '*' with '-' for range separation
    text = text.replace('*', '-')
    
    # Remove unwanted special characters while keeping alphanumeric characters, spaces, dots, and dashes
    text = re.sub(r'[^a-zA-Z0-9\s.\-\[\]]', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Convert to lowercase
    text = text.lower()
    print("combined_text : ", text)
    print("combined_text : ", text)
    matches = re.findall(pattern, text)
    unique_wattages = set()

    for match in matches:
        # Handle wattage ranges
        if match[0] and match[2]:
            watt_range = (float(match[0]), float(match[2]))
            if 'kw' in text:
                unique_wattages.add(f"[{watt_range[0]},{watt_range[1]}] kilowatt")
            else:
                unique_wattages.add(f"[{watt_range[0]},{watt_range[1]}] watt")
        
        # Single wattage value
        elif match[4]:
            wattage = float(match[4])
            if 'kw' in text:
                unique_wattages.add(f"{wattage} kilowatt")
            else:
                unique_wattages.add(f"{wattage} watt")
        
        # Handle units like 'HP' (horsepower)
        elif match[8]:
            wattage = float(match[8])
            unique_wattages.add(f"{wattage} horsepower")

    # Convert to list and sort to prioritize non-zero values
    sorted_wattages = sorted(unique_wattages, key=lambda x: (float(re.search(r'(\d+(\.\d+)?)', x).group()), x != "0.0 watt"), reverse=True)
    
    # Remove "0.0 watt" if present
    sorted_wattages = [w for w in sorted_wattages if w != "0.0 watt"]

    return sorted_wattages

def extract_item_weight(text):
    """
    Input: ocr text (str)
    Output: item weight (str). eg:- "5.0 gram" or ""
    """
    item_weight_units = {
        'gram': {'gram', 'g', 'gm', 'gr'},
        'kilogram': {'kilogram', 'kg'},
        'milligram': {'milligram', 'mg'},
        'ounce': {'ounce', 'oz'},
        'pound': {'pound', 'lbs', 'lb'},
        'ton': {'ton'}
    }

    def is_similar(word1, word2, threshold=0.7):
        similarity = difflib.SequenceMatcher(None, word1.lower(), word2.lower()).ratio()
        return similarity >= threshold
    
    # def convert_to_grams(value, unit):
    #     conversion_factors = {'microgram': 1e-6, 'milligram': 1e-3, 'gram': 1, 'kilogram': 1e3, 'ton': 1e6}
    #     return value * conversion_factors[unit]
    
    text = text.lower()
    text = text.replace("o", "0").replace("0z","oz").replace("iz", "12").replace("s", "5").replace(",", ".").replace("k9","kg").replace("1b","lb").replace("igr","1gr")
    text = re.sub(r'(\d)i', r'\1 1', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = ' '.join(text.split())
    
    pattern = r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+)'
    matches = re.findall(pattern, text)
    filtered_matches = []
    for match in matches:
        value = match[1]
        is_match = False
        for key in item_weight_units:
            if any(is_similar(value, unit_value) for unit_value in item_weight_units[key]):
                is_match = True
                break
        if is_match:
            if float(match[0]) < 0.1: continue
            filtered_matches.append((float(match[0]), key))
    matches = filtered_matches    

    if len(matches) == 0:
        return ""
    else:
        ounce_found = False
        for match in matches:
            if match[1] == 'ounce' or match[1] == 'pound':
                return f"{match[0]} {match[1]}"
        if not ounce_found:
            return f"{matches[0][0]} {matches[0][1]}"

image_url = 'https://m.media-amazon.com/images/I/71MbYieuadL.jpg'
text = process_image(image_url)

answer = extract_volume(text)
print(answer)  