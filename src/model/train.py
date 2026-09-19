import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.model.data import load_model_data

def train_logistic_regression():
    (
        x_train,
        x_validation,
        x_test,
        y_train,
        y_validation,
        y_test,
        feature_columns
    ) = load_model_data()

    pipeline = Pipeline(
        steps=[
            ("scaler",StandardScaler()),
            ("model",LogisticRegression(max_iter=1000,random_state=42))
        ]
    )

    pipeline.fit(x_train,y_train)

    return pipeline,x_validation,y_validation,x_test,y_test,feature_columns

if __name__ == "__main__":
    model, X_validation, y_validation, X_test, y_test, features = (
        train_logistic_regression()
    )

    print("Logistic Regression training completed successfully.")
    print(f"Number of features: {len(features)}")
    print(f"Validation predictions: {len(X_validation)}")
    print(f"Test predictions: {len(X_test)}")