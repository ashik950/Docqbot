import re
import json
import yaml
import spacy
from dateutil import parser

# Load the configuration file
with open(r"D:\Python_programs\version\PaddleOCR\config_bkc.yaml") as yaml_file:
    config = yaml.safe_load(yaml_file)

# Load key name mappings from config
key_name_mapping = config['Booking_Confirmation_Key_Name_Mapping']

class PostProcess:
    def __init__(self):
        # Initialize the spaCy model
        self.spacy_model = spacy.load('en_core_web_sm')

    def standardize_date_format(self, date_str):
        try:
            # Attempt to parse the date string with dateutil.parser
            parsed_date = parser.parse(date_str, dayfirst=True, yearfirst=False)
            return parsed_date.strftime("%d/%m/%Y")
        except (ValueError, TypeError):
            # Handle specific formats manually if needed
            if re.match(r'\d{2}\.\d{2}\.\d{4}', date_str):
                # Handle date format like '11.08.2021'
                return '/'.join(reversed(date_str.split('.')))
            elif re.match(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', date_str):
                # Handle date formats like '2023-7-12' or '2023/7/12'
                return parser.parse(date_str).strftime("%d/%m/%Y")
            else:
                print(f"Warning: Unable to parse the date '{date_str}'.")
                return ""

    def check_items(self, items):
        fields_compulsory = [
            "Shipper Name", "Shipper Address", "HBL_No", "Carrier Name", "Booking Number",
            "Departure Date", "Vessel Name", "Voyage No", "Country Code", "Port Code",
            "Port of Discharge", "Loading Terminal", "Gross Weight", "Gross Weight Unit",
            "Container number", "Container Size", "Number of Packages"
        ]

        available_keys_lower = {key.lower() for key in items.keys()}

        for key in fields_compulsory:
            if key.lower() not in available_keys_lower:
                items[key] = ""

        return items

    def change_keys(self, actual_data, key_map):
        if isinstance(actual_data, dict):
            return {key_map.get(key, key): self.change_keys(value, key_map) for key, value in actual_data.items()}
        elif isinstance(actual_data, list):
            return [self.change_keys(item, key_map) for item in actual_data]
        return actual_data

    def convert_values_to_strings(self, data):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    data[key] = str(value)
                elif isinstance(value, (dict, list)):
                    self.convert_values_to_strings(value)
        elif isinstance(data, list):
            for i in range(len(data)):
                if isinstance(data[i], (int, float)):
                    data[i] = str(data[i])
                elif isinstance(data[i], (dict, list)):
                    self.convert_values_to_strings(data[i])

    def post_process(self, input_data, key_name_mapping):
        # Ensure all required fields exist
        input_data = self.check_items(input_data)
        input_data['Departure Date'] = self.standardize_date_format(input_data['Departure Date'])
        # Apply key name mapping
        final_output = self.change_keys(input_data, key_name_mapping)

        # Convert numeric values to strings
        self.convert_values_to_strings(final_output)

        return final_output


# Example JSON data
json_part = {
  "Shipper Name": "CEVA LOGISTICS SINGAPORE PTE LTD",
  "Shipper Address": "80 ALPS AVENUE #01-01/02/03 MEZZANINE FLOOR SINGAPORE",
  "HBL_No": "",
  "Carrier Name": "",
  "Booking Number": "CMAUSIJ0340893",
  "Departure Date": "27 AUG 2021",
  "Vessel Name": "CMA CGM RACINE",
  "Voyage No": "0FF3PW1M",
  "Country Code": "IN",
  "Port Code": "NSA",
  "Port of Discharge": "INNSA",
  "Loading Terminal": "PASIR PANJANG TERMINAL",
  "Gross Weight": 11720,
  "Gross Weight Unit": "TNE",
  "Container number": "DFSU4339594",
  "Container Size": "40ST",
  "Number of Packages": 1
}

# Run post-processing
post_processor = PostProcess()
final_data = post_processor.post_process(json_part, key_name_mapping)

# Print the formatted output JSON
print(json.dumps(final_data, indent=4))
