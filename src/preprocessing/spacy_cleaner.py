import spacy

# Load spaCy English model once
nlp = spacy.load("en_core_web_sm")

def spacy_clean_text(text):
    doc = nlp(text.lower())  # lowercase here or inside pipeline
    tokens = [
        token.lemma_ for token in doc 
        if not token.is_stop and token.is_alpha
    ]
    return " ".join(tokens)
