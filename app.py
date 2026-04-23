# Import libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_FILE = "weather_forecast_data.csv"
TARGET_COLUMN = "Rain"


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return pd.read_csv(path)


def preprocess_data(data):
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' not found in dataset.")

    label_encoder = LabelEncoder()
    data[TARGET_COLUMN] = label_encoder.fit_transform(data[TARGET_COLUMN])

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]
    return X, y, label_encoder


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


def main():
    try:
        data = load_data(DATA_FILE)
    except FileNotFoundError as exc:
        print(exc)
        return

    print("Dataset Preview:")
    print(data.head())
    print("\nColumns:", list(data.columns))

    X, y, label_encoder = preprocess_data(data)
    model, X_test, y_test = train_model(X, y)
    evaluate_model(model, X_test, y_test, label_encoder)

    example_input = X.iloc[[0]]
    prediction = model.predict(example_input)

    print("\nExample input:")
    print(example_input.to_dict(orient="records")[0])
    print("Predicted Output:", label_encoder.inverse_transform(prediction)[0])


if __name__ == "__main__":
    main()