from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def build_vectorizer(texts, max_features=500):
    vectorizer = TfidfVectorizer(max_features=max_features)
    vectors = vectorizer.fit_transform(texts)
    return vectorizer, vectors

def save_vectorizer(vectorizer, path="models/tfidf_vectorizer.pkl"):
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)

def load_vectorizer(path="models/tfidf_vectorizer.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
