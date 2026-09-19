import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from src.model.data import load_model_data


def analyze_thresholds():
    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        feature_columns,
    ) = load_model_data()

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_validation)[:, 1]

    thresholds = [
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ]

    results = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        results.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y_validation,
                    predictions,
                ),
                "recall": recall_score(
                    y_validation,
                    predictions,
                ),
                "f1": f1_score(
                    y_validation,
                    predictions,
                ),
                "predicted_churn_rate": predictions.mean(),
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    results = analyze_thresholds()

    print("\nRandom Forest — Threshold Analysis")
    print("-" * 75)

    print(
        results.to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format,
                "predicted_churn_rate": "{:.4f}".format,
            },
        )
    )