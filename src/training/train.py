import json
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from evaluation import evaluate_classification

PROCESSED_JSON_PATH = "data/processed/cleaned_resumes.json"
MODEL_DIR = "models"

def load_data(path):
    import json

    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    data = []
    for item in raw_data:
        # Flatten nested lists
        if isinstance(item, list):
            data.extend(item)
        else:
            data.append(item)

    # Filter out items without label or cleaned_text
    filtered_data = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"Item {i} not dict.")
        if "cleaned_text" not in item or "label" not in item:
            print(f"Skipping item {i} missing 'cleaned_text' or 'label': {item}")
            continue
        filtered_data.append(item)

    texts = [item["cleaned_text"] for item in filtered_data]
    labels = [item["label"] for item in filtered_data]
    return texts, labels

def build_pipeline(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
        ("clf", clf)
    ])

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1

def main():
    texts, labels = load_data(PROCESSED_JSON_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    models = {
        "Logistic Regression": LogisticRegression(solver='liblinear', max_iter=1000, random_state=42),
        "SVM": SVC(kernel='linear', probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    os.makedirs(MODEL_DIR, exist_ok=True)

    best_f1 = 0
    best_model_name = None
    best_pipeline = None

    results = {}

    for name, clf in models.items():
        print(f"\nTraining {name}...")
        pipeline = build_pipeline(clf)

        with mlflow.start_run(run_name=name):
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            accuracy, precision, recall, f1 = evaluate_model(y_test, preds)

            mlflow.log_param("model", name)
            if hasattr(clf, "max_iter"):
                mlflow.log_param("max_iter", clf.max_iter)
            if hasattr(clf, "n_estimators"):
                mlflow.log_param("n_estimators", clf.n_estimators)
            if hasattr(clf, "kernel"):
                mlflow.log_param("kernel", clf.kernel)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            print(f"{name} → Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

            results[name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }

            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
                best_pipeline = pipeline

    # Save the best model
    if best_pipeline:
        save_path = os.path.join(MODEL_DIR, "best_model.joblib")
        joblib.dump(best_pipeline, save_path)
        print(f"\n✅ Best model '{best_model_name}' saved to '{save_path}'")

        # Final evaluation using evaluation.py
        final_preds = best_pipeline.predict(X_test)
        final_metrics = evaluate_classification(y_test, final_preds)
        print("\n📊 Detailed Evaluation of Best Model:")
        print(final_metrics)

    print("\n📋 Summary of All Models:")
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics}")

if __name__ == "__main__":
    main()
