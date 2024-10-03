import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_pump_data(start_date, num_days, sampling_interval_minutes=5):
    # Generate date range
    date_range = pd.date_range(start=start_date, periods=num_days * 24 * 60 // sampling_interval_minutes,
                               freq=f'{sampling_interval_minutes}min')

    # Base parameters
    base_vibration = 1.5
    base_temperature = 75
    base_pressure = 50
    base_flow_rate = 100
    base_power = 10
    base_rpm = 1800

    # Generate data
    data = {
        'timestamp': date_range,
        'vibration': np.random.normal(base_vibration, 0.2, len(date_range)),
        'temperature': np.random.normal(base_temperature, 2, len(date_range)),
        'pressure': np.random.normal(base_pressure, 1, len(date_range)),
        'flow_rate': np.random.normal(base_flow_rate, 3, len(date_range)),
        'power': np.random.normal(base_power, 0.5, len(date_range)),
        'rpm': np.random.normal(base_rpm, 20, len(date_range)),
    }

    df = pd.DataFrame(data)

    # Add some trends and anomalies
    df['vibration'] += np.linspace(0, 0.5, len(df))  # Gradual increase in vibration
    df['temperature'] += np.sin(np.linspace(0, 10 * np.pi, len(df))) * 3  # Cyclical temperature changes

    # Simulate a failure event
    failure_start = len(df) // 2
    failure_duration = 24 * 60 // sampling_interval_minutes  # 1 day
    df.loc[failure_start:failure_start + failure_duration, 'vibration'] *= 1.5
    df.loc[failure_start:failure_start + failure_duration, 'temperature'] += 10
    df.loc[failure_start:failure_start + failure_duration, 'flow_rate'] *= 0.9

    # Calculate hours since last maintenance
    df['hours_since_maintenance'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 3600

    # Add a failure flag
    df['failure'] = 0
    df.loc[failure_start:failure_start + failure_duration, 'failure'] = 1

    return df


# Generate 30 days of data
if __name__ == "__main__":
    start_date = datetime(2023, 1, 1)
    pump_data = generate_pump_data(start_date, num_days=30)
    pump_data.to_csv('../data/synthetic_pump_data.csv', index=False)
    print("Data saved to 'data/synthetic_pump_data.csv'")