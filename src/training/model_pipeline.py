from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_model_pipeline():
    """
    Builds a scikit-learn pipeline with TfidfVectorizer and LogisticRegression.
    
    Returns:
        sklearn.pipeline.Pipeline object
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,       # limit vocab size for efficiency
            ngram_range=(1, 2),      # use uni- and bi-grams
            stop_words='english'     # remove English stopwords
        )),
        ("clf", LogisticRegression(
            solver='liblinear',      # good for small datasets
            max_iter=1000,
            random_state=42
        ))
    ])
    return pipeline


if __name__ == "__main__":
    # Simple test
    model = build_model_pipeline()
    sample_texts = [
        "Experienced Python developer with knowledge in machine learning.",
        "Expert in Java and cloud platforms like AWS and Azure."
    ]
    sample_labels = [1, 0]  # Example binary labels

    model.fit(sample_texts, sample_labels)
    preds = model.predict(sample_texts)
    print("Predictions:", preds)
