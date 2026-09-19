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

from config.settings import (MODEL_FILE,TEST_DATA_FILE)
from src.model.load_model import load_model


def evaluate_final_model():
    artifact = load_model()

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    threshold = artifact["threshold"]

    test = pd.read_csv(
        TEST_DATA_FILE
    )

    X_test = test[feature_columns]
    y_test = test["churn"]

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    precision_curve, recall_curve, _ = precision_recall_curve(
        y_test,
        probabilities,
    )

    metrics = {
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
        "pr_auc": auc(
            recall_curve,
            precision_curve,
        ),
        "average_precision": average_precision_score(
            y_test,
            probabilities,
        ),
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
        ),
        "recall": recall_score(
            y_test,
            predictions,
        ),
        "f1": f1_score(
            y_test,
            predictions,
        ),
    }

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    return metrics, matrix, threshold


if __name__ == "__main__":
    metrics, matrix, threshold = evaluate_final_model()

    print("\nFinal Random Forest — Test Results")
    print("-" * 50)
    print(f"Classification threshold: {threshold}")

    for metric, value in metrics.items():
        print(f"{metric.upper():<20}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)