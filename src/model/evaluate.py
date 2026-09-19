import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.model.train_rf import train_random_forest

def evaluate_model(model,x,y):

    predictions = model.predict(x)
    probabilities = model.predict_proba(x)[:,1]

    precision, recall, _ = precision_recall_curve(y, probabilities)

    metrics = {
        "roc_auc": roc_auc_score(y, probabilities),
        "pr_auc": auc(recall, precision),
        "average_precision": average_precision_score(y, probabilities),
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions),
        "recall": recall_score(y, predictions),
        "f1": f1_score(y, predictions),
    }

    matrix = confusion_matrix(y,predictions)

    return metrics, matrix

if __name__ == "__main__":
    (
        model,
        X_validation,
        y_validation,
        X_test,
        y_test,
        features,
    ) = train_random_forest()

    metrics, matrix = evaluate_model(
        model,
        X_validation,
        y_validation,
    )

    print("\nRandom Forest — Validation Results")
    print("-" * 45)

    for metric, value in metrics.items():
        print(f"{metric.upper():<20}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)