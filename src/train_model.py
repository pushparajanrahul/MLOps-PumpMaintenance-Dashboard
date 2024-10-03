import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import prepare_data


def train_model():
    # Load data
    df = pd.read_csv('../data/synthetic_pump_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Prepare data
    X, y, scaler, feature_names = prepare_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Save model, scaler, and feature names
    joblib.dump(model, '../models/pump_failure_model.pkl')
    joblib.dump(scaler, '../models/scaler.pkl')
    joblib.dump(feature_names, '../models/feature_names.pkl')


if __name__ == "__main__":
    train_model()