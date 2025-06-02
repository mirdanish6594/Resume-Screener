from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_classification(y_true, y_pred, average='weighted'):
    """
    Compute common classification metrics: accuracy, precision, recall, F1-score.

    Args:
        y_true (list or array): True labels
        y_pred (list or array): Predicted labels
        average (str): Averaging method for multi-class metrics. Defaults to 'weighted'.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and f1_score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
