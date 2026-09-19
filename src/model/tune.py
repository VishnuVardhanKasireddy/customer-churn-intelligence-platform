import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.model.data import load_model_data


def evaluate_predictions(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)

    precision_curve, recall_curve, _ = precision_recall_curve(
        y_true,
        probabilities,
    )

    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": auc(recall_curve, precision_curve),
        "average_precision": average_precision_score(
            y_true,
            probabilities,
        ),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions),
        "recall": recall_score(y_true, predictions),
        "f1": f1_score(y_true, predictions),
    }


def tune_random_forest():
    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        feature_columns,
    ) = load_model_data()

    parameter_sets = [
        {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 300,
            "max_depth": 16,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 300,
            "max_depth": 20,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 300,
            "max_depth": 16,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
        },
    ]

    results = []

    for params in parameter_sets:
        model = RandomForestClassifier(
            **params,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_validation)[:, 1]

        metrics = evaluate_predictions(
            y_validation,
            probabilities,
        )

        results.append(
            {
                **params,
                **metrics,
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    results = tune_random_forest()

    print("\nRandom Forest Hyperparameter Comparison")
    print("-" * 70)

    print(
        results[
            [
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
                "max_features",
                "roc_auc",
                "pr_auc",
                "f1",
                "precision",
                "recall",
            ]
        ].to_string(index=False)
    )