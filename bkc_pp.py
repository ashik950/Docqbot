import re
import json
import yaml
import spacy
from dateutil import parser
# from embed_test import get_port_code  # Adjust path if modules are in different folders
from stdnum.iso6346 import is_valid
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
# Load the configuration file
with open(r"D:\Python_programs\version\PaddleOCR\config_bkc.yaml", "r", encoding="utf-8") as yaml_file:
    config = yaml.safe_load(yaml_file)

# Load key name mappings from config
key_name_mapping = config['Booking_Confirmation_Key_Name_Mapping']
uom_name_mapping = config['UOM_MAPPING']
class PostProcess:

    def __init__(self):
        # Initialize the spaCy model
        self.spacy_model = spacy.load('en_core_web_sm')
         
        # Reverse UOM Mapping Initialization
        self.reverse_uom_map = {}
        self.create_reverse_uom_map()
        self.fastapi_url = config.get('FASTAPI_URL', 'http://164.52.218.217:6090/generate')
        self.port_url =config.get('PORT_URL','http://216.48.186.19:8085/find-port')
        self.executor = ThreadPoolExecutor(max_workers=5)

    def create_reverse_uom_map(self):
        self.reverse_uom_map = {}
        uom_name_mapping = config.get('UOM_MAPPING', {})

        for code, names in uom_name_mapping.items():
            for name in names:
                if isinstance(name, str) and name.strip():  # ensure name is a non-empty string
                    self.reverse_uom_map[name.lower()] = code

    def extract_Booking_Number(self, text):
        if not isinstance(text, str):
            return text

        if ':' in text:
            return text.split(':', 1)[1].strip()

        return text.strip()

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
            "Shipper Name", "Shipper Address", "ConsigneeName", "HBL_No", "Carrier Name", "Booking Number",
            "Departure Date", "Vessel Name", "Voyage No", "Country Code", "Port Code",
            "Port of Discharge", "Loading Terminal", "Gross Weight", "Gross Weight Unit", "Container Shipment Mode",
            "Incoterms", "Container number", "Container Size", "Container Quantity", "Container Quantity Unit",
            "Outer Package", "Outer Package Unit"
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

    def validate_container_number(self, container_number: str) -> str:
        """Validates the container number according to the specified architecture."""
        if not container_number:
            return ""

        # Step 1: Remove spaces if present
        cleaned_number = container_number.replace(" ", "")

        # Step 2: Extract first 11 characters if more than 11 characters
        if len(cleaned_number) > 11:
            cleaned_number = cleaned_number[:11]

        # Step 3: Validate the 11-character string
        return cleaned_number if is_valid(cleaned_number) else ""

    def extract_container_size(self, container_size):
        # Extract numeric part of the container size
        match = re.match(r'(\d+)', container_size)
        return match.group(1) if match else ""

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

    def standardize_uom(self, uom):
        """Standardizes the UOM to its code based on reverse_uom_map."""
        return self.reverse_uom_map.get(uom.lower(), uom)
    
    def fetch_port_code(self, checked_items):
        try:
            # print("entering fetch_port_code")
            # port_code=""
            # msg_type = checked_items.get("MsgType", "")
            # if msg_type == "":
            #     return ""
            # elif(msg_type == "Import"):
            #     port_description = checked_items.get("Port of Loading", "")
            #     country_code = checked_items.get("Shipper Country Code", "")
            # elif(msg_type == "Export"):
            #     port_description = checked_items.get("Port of Discharge", "")
            #     country_code = checked_items.get("Consignee Country Code", "")
            # if(country_code=="SG" and not  port_description):
            #     print("The port code is: SGSIN")
            #     return "SGSIN"
            # else:
                print("API call")
                country_code = checked_items.get("Country Code", "")
                port_of_discharge = checked_items.get("Port of Discharge", "")
                response = requests.post(self.port_url, json={"user_description": port_of_discharge, "country_code": country_code})
                print("API response",response)
                response.raise_for_status()
                return response.json().get("port_code", "")
        except requests.RequestException as e:
            print(f"Error fetching HS code: {e}")
            return "" 
    def post_process(self, input_data, key_name_mapping):
        # Ensure all required fields exist
        input_data = self.check_items(input_data)

        # Standardize departure date format
        input_data['Departure Date'] = self.standardize_date_format(input_data['Departure Date'])
        input_data["Port of Discharge"] = self.fetch_port_code(input_data) 
        # Get Port Code using embedding function
        country_code = input_data.get("Country Code", "")
        # location_name = input_data.get("Port of Discharge", "")
        port_code = ""
        # if country_code and location_name:
        #     port_code = get_port_code(country_code, location_name)
        #     input_data["Port Code"] = port_code
        # else:
        #     input_data["Port Code"] = ""

        # Update 'Port of Discharge' to "CountryCode + PortCode"
        if country_code and port_code:
            input_data["Port of Discharge"] = f"{country_code.upper()}{port_code.upper()}"

        # Convert Gross Weight to metric tons if it's a number
        if "Gross Weight" in input_data and isinstance(input_data["Gross Weight"], (int, float)):
            input_data["Gross Weight"] = input_data["Gross Weight"] / 1000

        # Extract numeric part of container size
        if "Container Size" in input_data:
            input_data["Container Size"] = self.extract_container_size(input_data["Container Size"])

        # Clean OBL number to remove prefixes like "UCR NO:"
        # Extract clean OBL number
        if "Booking Number" in input_data:
            input_data["Booking Number"] = self.extract_Booking_Number(input_data["Booking Number"])

        if "Container number" in input_data:
            input_data["Container number"] = self.validate_container_number(input_data.get("Container number", ""))

        # Standardize Outer Package Unit (UOM)
        if "Outer Package Unit" in input_data:
            input_data["Outer Package Unit"] = self.standardize_uom(input_data["Outer Package Unit"])

        # Inferred Container Type based on Shipment Mode
        shipment_mode = input_data.get("Container Shipment Mode", "").upper().strip()
        inferred_map = {
            "CY/CY": "FCL",
            "CFS/CFS": "LCL",
            "CY/CFS": "LCL",
            "CFS/CY": "FCL"
        }
        inferred_container_type = inferred_map.get(shipment_mode, "")
        if inferred_container_type:
            input_data['Container Shipment Mode'] = inferred_container_type

        # ContainerSizeType logic
        container_size = input_data.get("Container Size", "")
        if input_data['Container Shipment Mode'] and container_size:
            input_data["ContainerSizeType"] = f"{input_data['Container Shipment Mode'].upper()}{container_size.upper()}"

        fields_compulsory = ["Country Code", "Port Code", "Container Size"]

        for item in fields_compulsory:
            input_data.pop(item, None)  # Safely remove item if it exists

        final_output = self.change_keys(input_data, key_name_mapping)

        # Convert all numeric values to strings for uniformity
        self.convert_values_to_strings(final_output)

        return final_output


# Example JSON data
json_part = {
  "Shipper Name": "",
  "Shipper Address": "",
  "Consignee Name": "",
  "HBL_No": "",
  "Carrier Name": "MAERSK",
  "Booking Number": "212437194",
  "Departure Date": "14 Aug 2021",
  "Vessel Name": "COSCO YINGKOU",
  "Voyage No": "131W",
  "Port of Discharge": "THE ROAD",
  "Country Code": "AI",
  "Loading Terminal": "PSA Singapore Terminal",
  "Gross Weight": 18000.000,
  "Gross Weight Unit": "TNE",
  "Container number": "",
  "Container Size": "40 DRY",
  "Container Shipment Mode":  "FCL",
  "Container Quantity": 1,
  "Container Quantity Unit": "UNT",
  "Outer Package": 1,
  "Outer Package Unit": "Piece(s)",
  "Incoterm": ""
}

# Run post-processing
post_processor = PostProcess()
final_data = post_processor.post_process(json_part, key_name_mapping)

# Print the formatted output JSON
print(json.dumps(final_data, indent=4))
