import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def engineer_features(df):
    # Check if df is a single row (Series) and convert it to a DataFrame if so
    if isinstance(df, pd.Series):
        df = df.to_frame().T

    # Create rolling statistics
    for feature in ['vibration', 'temperature', 'pressure', 'flow_rate', 'power', 'rpm']:
        if len(df) > 1:
            df[f'{feature}_rolling_mean_1h'] = df[feature].rolling(window=12, min_periods=1).mean()
            df[f'{feature}_rolling_std_1h'] = df[feature].rolling(window=12, min_periods=1).std()
        else:
            # For single row, just use the value itself
            df[f'{feature}_rolling_mean_1h'] = df[feature]
            df[f'{feature}_rolling_std_1h'] = 0  # Standard deviation of a single value is 0

    # Calculate rate of change
    for feature in ['vibration', 'temperature', 'pressure', 'flow_rate', 'power', 'rpm']:
        if len(df) > 1:
            df[f'{feature}_rate_of_change'] = df[feature].diff() / df['timestamp'].diff().dt.total_seconds()
        else:
            df[f'{feature}_rate_of_change'] = 0  # Rate of change for a single value is 0

    # Fill NaN values with 0 for single-row inputs
    df = df.fillna(0)

    return df


def prepare_data(df):
    # Engineer features
    df = engineer_features(df)

    # Prepare features and target
    features = [col for col in df.columns if col not in ['timestamp', 'failure']]
    X = df[features]
    y = df['failure']

    # Save feature names
    feature_names = X.columns.tolist()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, feature_names