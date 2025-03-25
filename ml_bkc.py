import os
import paddleocr
import torch
from pdf2image import convert_from_path
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv

def extract_text_from_pdf(pdf_path, output_folder, api_key, model):
    """
    Extracts text from a PDF using PaddleOCR, returning the combined text
    with proper newlines and handling potential errors. Saves each page's text
    as a separate text file in the specified output folder.
    Args:
        pdf_path (str): Path to the PDF file.
        output_folder (str):   to the folder where text files will be saved.
        api_key (str): Mistral AI API key.
        model (str): Mistral AI model name.
    Returns:
        str: Success message if text extraction is successful, otherwise an error message.
    """
    config = {
            "use_gpu": torch.cuda.is_available(),
            "lang": "en",
            # Detection Module Configuration
            "det": {
                "architecture": "DBNet++",
                "pre_trained": True,
                "model_dir": "./models/dbnet_v3_det/",
                "use_gpu": torch.cuda.is_available(),
                "backbone": "ResNet50_vd",
            },
            # Recognition Module Configuration
            "rec": {
                "architecture": "SAR",
                "character_type": "alphanumeric",
                "num_classes": 95,
                "use_gpu": torch.cuda.is_available(),
                "model_dir": "./models/sar_v3_rec/",
                "transform": {
                    "resize": (64, 256),
                    "normalize": True
                }
            },
            # Additional Recognition Settings
            "min_size_for_recognize": 15,
            "det_db_thresh": 0.5,
            "det_char_conf": 0.4,
            "cls_conf_thresh": 0.7,
            "det_limit_type": "max",
            "use_angle_cls": True,
            "max_text_length": 30000,
            # Layout and Deskewing Enhancements
            "layout": True,
            "deskew": True,
            "rotation_aware": True,
            # Post-processing and Optimization
            "postprocess": {
                "merge_lines": True,
                "min_text_score": 0.6,
                "text_filtering": "advanced",
                "text_merge_strategy": "smart",
            },
            # System Optimization
            "precision": "mixed",
            "enable_mkldnn": not torch.cuda.is_available(),
            "batch_size": 16,
            "enable_async": True,
            # Logging and Debugging Options
            "log_interval": 50,
            "save_model": True,
        }
    try:
        ocr = paddleocr.PaddleOCR(**config)
        results = ocr.ocr(pdf_path, cls=True)
        client = MistralClient(api_key=api_key)
        pages = convert_from_path(pdf_path, dpi=600)
        for i, page in enumerate(results):
            page_text = ""
            for line in page:
                text = line[1][0]  # Access the text content of the line
                page_text += f"{text}\n"
               # print(text)

            # Save page text to a text file
            page_filename = f"page_{i+1}.txt"
            output_path = os.path.join(output_folder, page_filename)
            with open(output_path, "w", encoding="utf-8") as text_file:
                text_file.write(page_text.strip())

            # Define the user prompt
            user_prompt = '''
 You are a Booking Confirmation Data Extractor responsible for extracting specific fields as mentioned from the booking confirmation text provided. Extract the required fields and convert them into JSON format. Include all rows, even if fields are missing (set them to empty strings in the JSON). Strictly follow these conditions: return only the JSON output, with no extra text, comments, notes, or instructions.
  Special Condition: If a booking confirmation is spread across multiple pages, ensure that keywords and associated values found on the first page are considered when processing subsequent pages. For example, if certain keywords are found on the first page but their corresponding values are located on the second page, extract the values as if the context was continuous across pages.
  
  1. Shipper Name: Extract the full shipper's name by identifying the company name preceding the shipper's address. Prioritize explicitly mentioned keywords like 'SHIP FROM', 'FROM', 'From/via', 'Shipper' or 'Exporter.' If these keywords are not present, extract the first identifiable company name before the shipper's address. Ensure the name includes valid suffixes such as 'Pte. Ltd.,' 'Singapore Pte. Ltd.,' 'Services Limited,' while excluding any location or address details.
  2. Shipper Address: Extract the full shipper's address that directly follows the shipper’s name. Ensure only the address details are captured, excluding company names and unrelated sections like 'Delivery Address' or 'Invoice Address.' If no direct keyword is present, extract the address associated with the identified shipper name.
  3  HBL_No : Identify the number from the keyword "House" or "HBL NO".
  3  Carrier Name:  Identify and extract the carrier name only from the keyword "Carrier". If the keyword is not found, return an empty string.
  4  Booking Number: A unique numeric or alphanumeric code assigned by the shipping company to identify the booking.Identify and extract the "Booking Number" number by identifying common terms such as "OBL","Booking Number","Booking_confirmation"
  5  Departure Date: "Extract the primary Estimated Time of Departure (ETD) date from the document, focusing on the first international departure from Singapore. Look for key phrases such as 'Estimated Departure', 'ETD', 'Departure date', or similar terms. Ensure the extracted date is in the format DD MMM YYYY (day as two digits, month as three letters, and year as four digits). Exclude time (hours and minutes). If multiple ETD dates are found, prioritize the one associated with the first international leg of the journey from Singapore. If no ETD date is found, return an empty string."
  6  Vessel Name: The name of the vessel on which the cargo is to be transported.
  7  Voyage Number: A unique identifier for the specific voyage of the vessel.
  8  Country Code: Country codes are standardized by the International Organization for Standardization (ISO) under the ISO 3166-1 alpha-2 standard. These codes are two-letter identifiers for countries. 
  For example:
  CN represents China.
  US represents the United States.
  GP represents Guam.
  9  Port Code: What is the three-letter UN/LOCODE seaport code for "Port Of Discharge:" or "Port Delivery"?. Provide only the code without any explanation. Example: If the data includes "Final Destination: LOS ANGELES, CA, USA", the Port Code is LAX.
  10 Loading Terminal: The terminal or specific location at the port of loading where the cargo will be loaded onto the vessel.
  11 Total Gross Weight: Uncover and collect the entire gross weight stated in the shipment through the keyword "Gross Weight."
  12 Total Gross Weight Unit: Default the unit to "TNE".Strictly ignore any other Units
  13 Container number: Extract the Container Number from the provided text. A valid container number follows the ISO 6346 standard and consists of:
    (i)   Owner Prefix Code: The first three capital letters identifying the owner (e.g., "MAEU").
    (ii)  Equipment Category Identifier: The fourth character (e.g., "U" for freight containers).
    (iii) Serial Number: A six-digit numeric code (e.g., "123456").
    (iv)  Check Digit: The final digit, typically boxed or separated, used for validation (e.g., "7").
  Example:
  "MAEU1234567" → Owner: MAEU, Category: U, Serial: 123456, Check Digit: 7
  "CAXU7891234" → Owner: CAX, Category: U, Serial: 789123, Check Digit: 4
  If the container number is invalid or missing, return an empty string. Validate against the above format, ensuring all components are present and in order.
  14 Container Size: Extract the Container Size from the document using keywords such as "Size" or "CNTR SIZE." Look for values like "40RK," "20GP," or other similar alphanumeric codes that indicate the size and type of the container(e.g., "GP", "RK", "SD", "HC", "SOC"). Ensure the extracted value strictly matches this format and type. If no container size is found, return an empty string.
  15 No of packages: Extract the Number of Packages from the booking confirmation. This field typically indicates the total count of packages or containers listed for shipment. Return the value as an integer or float. If no such field is found, return an empty string.
  Expected JSON output format:
  {
    "Shipper Name": "{{"type":"string"}}",
    "Shipper Address": "{{"type":"string"}}",
    "HBL_No":"{{"type":"string"}}",
    "Carrier Name": "{{"type":"string"}}",
    "Booking Number": "{{"type":"string"}}",
    "Departure Date": [{{"type": "string", "format": "date"}}],
    "Vessel Name": "{{"type":"string"}}",
    "Voyage No": "{{"type":"string"}}",
    "Country Code": "{{"type":"string"}}",
    "Port Code":  "{{"type":"string"}}",
    "Port of Discharge": Country Code+Port code,
    "Loading Terminal": "{{"type":"string"}}",
    "Gross Weight": "{{"type":"number"}}",
    "Gross Weight Unit":"{{"type":"string"}}",
    "Container number":"{{"type":"string"}}",
    "Container Size":"{{"type":"string"}}",
    "Number of Packages":"{{"type":"number"}}"
  }
  
    Never say "continue with the remaining," please produce all the tokens available in the input text.
    Please ensure that if any of the fields are not present in the text, they are set to empty values in the JSON response.
    Present only the JSON output, without any additional text, comments, or sequential instructions.
  
    ###
    <<<  
     Context:
     {text}
     Answer: '''

            chat_response = client.chat(
                model=model,
                messages=[
                    ChatMessage(role="user",stream=True, content=user_prompt + "Context:\n" + page_text)
                ]
            )
            extracted_text = chat_response.choices[0].message.content
            print(extracted_text)

            usage_info = chat_response.usage
            print("Usage info:", usage_info)
            if usage_info:
                prompt_tokens = usage_info.prompt_tokens
                total_tokens = usage_info.total_tokens
                completion_tokens = usage_info.completion_tokens
                print(f"Prompt Tokens: {prompt_tokens}")
                print(f"Total Tokens: {total_tokens}")
                print(f"Completion Tokens: {completion_tokens}")
                
            output_filename = f"mistral_response_{i+1}.json"
            output_path = os.path.join(output_folder, output_filename)
            
            with open(output_path, "w", encoding="utf-8") as json_file:
                json_file.write(extracted_text)
                

            return "Text extraction completed. Text files saved in output folder."
    except Exception as e:
        error_message = f"Error extracting text from PDF: {e}"
        print(error_message)
        return error_message
    

# Example usage
if __name__ == "__main__":
    pdf_path =r"D:\BOOKINGCONFIRMATION\BOOKINGCONFIRMATION\183178192_page_1.pdf"
    output_folder = "D:\\Python_programs\\output"  # Change this to your desired output folder
    print(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Add your Mistral AI API key and model name here
    dotenv_path = r'F:\\chatbot\\memory\\chatbot\\enke.env'  # Specify the path to your key.env file

    # Load environment variables from the specified .env file
    load_dotenv(dotenv_path)

    # Access the environment variable and print it
    api_key = os.getenv("MY_KEY")
    model = "mistral-large-latest"

    result = extract_text_from_pdf(pdf_path, output_folder, api_key,model)
    print(result)
