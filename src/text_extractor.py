import pytesseract
from PIL import Image

class TextExtractor:
    def __init__(self, tesseract_cmd):
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def extract_text_from_image(self, image_path):
        image = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(image)
        cleaned_text = extracted_text.strip()
        return cleaned_text

# Example usage in another script:
# from text_extractor import TextExtractor
# extractor = TextExtractor(r'C:\Program Files\Tesseract-OCR\tesseract.exe')
# extracted_text = extractor.extract_text_from_image("img.jpg")
# print("Extracted Text:")
# print(extracted_text)
