import os
from docx import Document
from pdfminer.high_level import extract_text


def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)


def load_resumes(directory):
    resumes = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(filepath)
        elif filename.endswith(".docx"):
            text = extract_text_from_docx(filepath)
        else:
            continue
        resumes.append({"filename": filename, "content": text})
    return resumes
