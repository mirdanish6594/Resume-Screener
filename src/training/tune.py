from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import json
import os

# Load processed data function (you can adapt as needed)
def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [item['cleaned_text'] for item in data]
    labels = [item['label'] for item in data]  # Assuming you have labels here
    return texts, labels

def main():
    DATA_PATH = "data/processed/cleaned_resumes.json"

    X, y = load_data(DATA_PATH)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=500))
    ])

    param_grid = {
        'tfidf__max_df': [0.75, 1.0],
        'tfidf__ngram_range': [(1,1), (1,2)],
        'clf__C': [0.1, 1, 10],
        'clf__solver': ['liblinear']  # good default for small datasets
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='f1_weighted',
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X, y)

    print("Best params:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)

    # Save best model if needed
    import joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(grid_search.best_estimator_, "models/best_logreg_model.joblib")

if __name__ == "__main__":
    main()
