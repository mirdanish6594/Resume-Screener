import nltk
nltk.data.path.append('/home/mirdanish/nltk_data')

import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    cleaned = [word for word in tokens if word not in stop_words and word.isalpha()]
    return " ".join(cleaned)

# Simple skill matcher
SKILLS = [
    "python", "java", "sql", "tensorflow", "keras", "pytorch", "docker", "kubernetes",
    "aws", "azure", "nlp", "opencv", "machine learning", "deep learning", "flask", "git"
]

def extract_skills(text):
    text = text.lower()
    found = [skill for skill in SKILLS if skill in text]
    return list(set(found))
