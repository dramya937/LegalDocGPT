import os
from utils.text_cleaner import clean_text
from PyPDF2 import PdfReader
import docx

def parse_contract(file_path):
    text = ""
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        raise ValueError("Unsupported file type. Please use PDF or DOCX.")
    
    return clean_text(text)
