import pandas as pd

from src.model.load_model import load_model
from src.model.train_rf import train_random_forest


def get_feature_importance(model, feature_columns):
    importance = pd.DataFrame({
        "feature": list(feature_columns),
        "importance": model.feature_importances_,
    })

    return importance.sort_values(
        by="importance",
        ascending=False,
    ).reset_index(drop=True)


def get_saved_model_importance():
    artifact = load_model()

    return get_feature_importance(
        artifact["model"],
        artifact["feature_columns"],
    )


if __name__ == "__main__":
    (
        model,
        X_validation,
        y_validation,
        X_test,
        y_test,
        feature_columns,
    ) = train_random_forest()

    importance = get_feature_importance(
        model,
        feature_columns,
    )

    print("\nRandom Forest — Feature Importance")
    print("-" * 45)

    for _, row in importance.iterrows():
        print(f"{row['feature']:<30} {row['importance']:.4f}")