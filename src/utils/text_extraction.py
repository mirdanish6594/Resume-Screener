import fitz  # PyMuPDF
from pdfminer.high_level import extract_text as pdfminer_extract_text
from docx import Document
import textract
import os

def extract_text_from_pdf_pymupdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_pdf_pdfminer(pdf_path):
    return pdfminer_extract_text(pdf_path)

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    full_text = [para.text for para in doc.paragraphs]
    return '\n'.join(full_text)

def extract_text_from_doc(doc_path):
    try:
        text = textract.process(doc_path).decode("utf-8")
        return text
    except Exception as e:
        raise RuntimeError(f"Error extracting from DOC file: {e}")

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        return extract_text_from_pdf_pymupdf(file_path)
        # or use: return extract_text_from_pdf_pdfminer(file_path)
    
    elif ext == '.docx':
        return extract_text_from_docx(file_path)
    
    elif ext == '.doc':
        return extract_text_from_doc(file_path)
    
    elif ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    else:
        raise ValueError(f"Unsupported file format: {ext}")
