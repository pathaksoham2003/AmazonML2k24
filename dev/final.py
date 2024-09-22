import re
import requests
from PIL import Image,UnidentifiedImageError
from io import BytesIO
import easyocr
import numpy as np
import pandas as pd
import difflib
import json
import os
## 131118 images
batches = {
    1:[0,10000],
    2:[10000,20000],
    3:[20000,30000],
    4:[30000,40000],
    5:[50000,60000],
    6:[60000,70000],
    7:[80000,90000],
    8:[90000,100000],
    9:[100000,110000],
    10:[110000,120000],
    11:[120000,130000],
    12:[130000,131119],
}
checkpoint_file = "checkpoint.json"
def load_checkpoint():
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as file:
            data = json.load(file)
            return data.get("batch", 1), data.get("index", 0)
    return None, None

# Function to save the current batch and index to the checkpoint file
def save_checkpoint(current_batch, current_index):
    with open(checkpoint_file, 'w') as file:
        json.dump({"batch": current_batch, "index": current_index}, file)


reader = easyocr.Reader(['en'])

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
    
    return list(unique_volts)[0]  


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
    
    return list(unique_volumes)[0]

def clean_text(text):
    text = text.replace('O', '0').replace('S', '5').replace("IZ", "12").replace(",",".").replace("I","1")
    
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = ' '.join(text.split())
    text = text.lower()
    return text

def extract_largest_height(text):
    # Updated regex pattern to capture heights
    pattern = r'(\d+(\.\d+)?)[\s]*-[\s]*(\d+(\.\d+)?)[\s]*(cm|mm|inches?|in|ft|\'|"|m|meter|millimeter|yard|yd)\b|\b(\d+(\.\d+)?)[\s]*(cm|mm|inches?|in|ft|\'|"|m|meter|millimeter|yard|yd)\b'

    # Allowed units with their standard format and conversion to a common base (cm for simplicity)
    unit_mapping = {
        'cm': ('centimetre', 1),  # Conversion factor to centimetres
        'centimeter': ('centimetre', 1),
        'm': ('metre', 100),
        'meter': ('metre', 100),
        'mm': ('millimetre', 0.1),
        'millimeter': ('millimetre', 0.1),
        'in': ('inch', 2.54),
        'inches': ('inch', 2.54),
        '"': ('inch', 2.54),
        'ft': ('foot', 30.48),
        "'": ('foot', 30.48),
        'yard': ('yard', 91.44),
        'yd': ('yard', 91.44)
    }

    # Search for matches in the text
    matches = re.findall(pattern, text)
    
    heights = []  # List to store height values
    
    for match in matches:
        # Handle height ranges like '150-175 cm'
        if match[0] and match[2]:
            unit = unit_mapping.get(match[4], None)
            if unit:
                # Convert both values in the range to cm, take the larger one
                height1 = float(match[0]) * unit[1]
                height2 = float(match[2]) * unit[1]
                heights.append((max(height1, height2), unit[0]))
        # Handle single height values
        elif match[5]:
            unit = unit_mapping.get(match[7], None)
            if unit:
                height = float(match[5]) * unit[1]
                heights.append((height, unit[0]))
    
    # Find the largest height
    if heights:
        largest_height = max(heights, key=lambda x: x[0])  # Get the largest height
        # Convert back to the original unit
        for key, value in unit_mapping.items():
            if value[0] == largest_height[1]:
                original_unit = key
                break
        else:
            original_unit = largest_height[1]  # Fallback if not found

        return f"{largest_height[0] / unit_mapping[original_unit][1]:.1f} {unit_mapping[original_unit][0]}"
    else:
        return None  # No height found

def extract_wattage(text):
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
    
    # Replace common OCR errors and fix known issues
    pattern = r'(\d+(\.\d+)?)[\s]*-[\s]*(\d+(\.\d+)?)[\s]*(?:[kK][wW]|[wW](?:atts?)?|[hH][pP])\b|' \
              r'\b(\d+(\.\d+)?)[\s]*(?:[kK][wW]|[wW](?:atts?)?|[hH][pP])\b|' \
              r'\b[kK][wW](\d+(\.\d+)?)\b|' \
              r'\b[wW](\d+(\.\d+)?)\b|' \
              r'(\d+(\.\d+)?)[\s]*(?:[kK][wW]|[wW](?:atts?)?)|' \
              r'\b(\d*)OOW\b'
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

    return sorted_wattages[0]

def extract_item_weight(text,isMax):
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
    
    def convert_to_grams(value, unit):
        conversion_factors = {'microgram': 1e-6, 'milligram': 1e-3, 'gram': 1, 'kilogram': 1e3, 'ton': 1e6}
        return value * conversion_factors[unit]
    
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
    
    if isMax:
        return max(matches, key=lambda x: convert_to_grams(x[0], x[1]))
    
    if len(matches) == 0:
        return ""
    else:
        ounce_found = False
        for match in matches:
            if match[1] == 'ounce' or match[1] == 'pound':
                return f"{match[0]} {match[1]}"
        if not ounce_found:
            return f"{matches[0][0]} {matches[0][1]}"

def extract_depth_width(text, maximum=True):
    """
    Input: ocr text (str)
    Output: depth/width (str). eg:- "5.0 cm" or ""
    """
    depth_units = {
        'centimetre': {'centimetre', 'cm', 'centimeter', 'centimeters', 'cms', 'cmeter', 'cmeters', 'cmtr', 'cmtrs', 'c'},
        'foot': {'foot', 'ft', 'feet', 'fts', 'ftt', 'ftts'},
        'inch': {'inch', 'in', '"', '”', '″'},
        'metre': {'metre', 'm', 'meter', 'meters', 'mtr', 'mtrs', 'mt', 'mts', 'metres'},
        'millimetre': {'millimetre', 'mm', 'millimeter', 'millimeters', 'mmeter', 'mmeters', 'mmtr', 'mmtrs', 'mmtr', 'mmtrs'},
        'yard': {'yard', 'yd', 'yds', 'yrds', 'yards'},
        'nanometre': {'nanometre', 'nm', 'nanometer', 'nanometers', 'nmeter', 'nmeters', 'nmtr', 'nmtrs', 'nmtr', 'nmtrs'},
        'kilometre': {'kilometre', 'km', 'kilometer', 'kilometers', 'kmeter', 'kmeters', 'kmtr', 'kmtrs', 'kmtr', 'kmtrs'},
        'mil': {'mil', 'milimeter', 'milimeters', 'milimetre', 'milimetres', 'milimeter', 'milimeters', 'milimetre', 'milimetres'},
        'micrometre': {'micrometre', 'µm', 'micrometer', 'micrometers', 'µmeter', 'µmeters', 'µmtr', 'µmtrs', 'µmtr', 'µmtrs'},
        'thou': {'thou', 'thousandth', 'thousandths', 'thousand', 'thousands', 'th'},
    }

    def is_similar(word1, word2, threshold=0.7):
        similarity = difflib.SequenceMatcher(None, word1.lower(), word2.lower()).ratio()
        return similarity >= threshold
    
    text = text.lower()
    text = text.replace("o", "0").replace("iz", "12").replace(",", ".").replace("k9","kg").replace("1b","lb")
    text = re.sub(r'(\d)i', r'\1 1', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = ' '.join(text.split())
    
    pattern = r'(\d+(?:\.\d+)?)\s*([a-zA-Z"”″]+)'
    matches = re.findall(pattern, text)
    filtered_matches = []
    for match in matches:
        value = match[1]
        is_match = False
        for key in depth_units:
            if any(is_similar(value, unit_value) for unit_value in depth_units[key]):
                is_match = True
                break
        if is_match:
            if float(match[0]) < 0.1: continue
            filtered_matches.append((float(match[0]), key))
    matches = filtered_matches    

    if len(matches) == 0:
        return ""
    else:
        if maximum:
            max_value = max(matches, key=lambda x: x[0])
            return f"{max_value[0]} {max_value[1]}"
        else:
            min_value = min(matches, key=lambda x: x[0])
            return f"{min_value[0]} {min_value[1]}"

def process_image(image_url):
    try:
        response = requests.get(image_url, timeout=10)  # Add timeout for image request
        if response.status_code == 200:  # Check if the response is successful
            try:
                image = Image.open(BytesIO(response.content))
                return image
            except UnidentifiedImageError:
                print(f"Unable to identify image from URL: {image_url}")
                return None
        else:
            print(f"Failed to retrieve image from URL: {image_url}")
            return None
    except Exception as e:
        return None
    

def process_height_image(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    
    reader = easyocr.Reader(['en'])
    
    rotations = [90, 270, 0]  # Rotations to process
    
    for angle in rotations:
        rotated_image = image.rotate(angle, expand=True)
        
        rotated_image_np = np.array(rotated_image)
        result = reader.readtext(rotated_image_np)
        
        
        text = ' '.join([res[1] for res in result])
        cleaned_text = clean_text(text) 
        print(f"{angle}= {cleaned_text}")
        predicted_value = extract_largest_height(cleaned_text)
        if predicted_value:
            return predicted_value # Stop after the first valid prediction
        
        # Store OCR result for this rotation
        
    
    return None  # Return all OCR results
##  

def update_dataframe(df, start_index, end_index, file_name,batch):
    # Ensure that the range is within the bounds of the DataFrame
    end_index = min(end_index, len(df))

    # Iterate over the specified range
    for i in range(start_index, end_index):
        image_link = df.loc[i, 'image_link']
        entity_name = df.loc[i, 'entity_name']
        save_checkpoint(batch, i)
        print(i)
        image = process_image(image_link)
        ocr_text,result = None,None
        text = None
        if image is not None:
            if entity_name != "height":
               result = reader.readtext(image)  
               text = ' '.join([res[1] for res in result])
               ocr_text = text  
               result = process_height_image(image_link)
               print(str(result))
            else:
                result = reader.readtext(image)  
                text = ' '.join([res[1] for res in result])
                ocr_text = text            
                if entity_name == "wattage":
                    result = extract_wattage(text)
                elif entity_name == "voltage":
                    result = extract_volts(text)
                elif entity_name == "item_weight":
                    result = extract_item_weight(text,False)
                elif entity_name == "maximum_weight_recommendation":
                    result = extract_item_weight(text,True)
                elif entity_name == "width":
                    result = extract_depth_width(text,True)
                elif entity_name == "depth":
                    result = extract_depth_width(text,False)
                else:
                    result = ""           
        if ocr_text is None:
            ocr_text = ""
        if result is None:
            result = ""
            
        if 'ocr_text' not in df.columns:
            print(ocr_text)
            df['ocr_text'] = ocr_text
        if 'result' not in df.columns:
            df['result'] = result
        df.loc[i, 'ocr_text'] = ocr_text
        df.loc[i, 'result'] = result
        df.to_csv(file_name, index=False)

df = pd.read_csv("./test.csv")

current_batch, current_index = load_checkpoint()
# Enter the batch that you are processing
BATCH = 1
start = batches[BATCH][0]
if current_batch is not None:
    BATCH = current_batch
if current_index is not None:
     start = current_index - 1
   
update_dataframe(df, 3, batches[BATCH][1], f'test{BATCH}.csv',BATCH)

