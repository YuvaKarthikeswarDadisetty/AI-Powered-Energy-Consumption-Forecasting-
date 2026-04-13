# ==============================
# Forecast + Future Prediction
# ==============================

import pandas as pd
import matplotlib.pyplot as plt
import joblib


def load_data():
    df = pd.read_csv('data/processed_energy.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def load_model():
    return joblib.load('models/energy_model.pkl')


def predict_existing(df, model):
    X = df[['hour', 'day', 'month', 'day_of_week']]
    df['predicted_energy'] = model.predict(X)
    return df


# 🔥 NEW: FUTURE PREDICTION
def predict_future(model, last_timestamp, hours=24):
    future_dates = pd.date_range(start=last_timestamp, periods=hours+1, freq='H')[1:]

    future_df = pd.DataFrame({'timestamp': future_dates})

    # Feature engineering
    future_df['hour'] = future_df['timestamp'].dt.hour
    future_df['day'] = future_df['timestamp'].dt.day
    future_df['month'] = future_df['timestamp'].dt.month
    future_df['day_of_week'] = future_df['timestamp'].dt.dayofweek

    X_future = future_df[['hour', 'day', 'month', 'day_of_week']]

    future_df['predicted_energy'] = model.predict(X_future)

    return future_df


def plot_all(df, future_df):
    plt.figure(figsize=(12,6))

    # Historical
    plt.plot(df['timestamp'], df['energy'], label='Actual')
    plt.plot(df['timestamp'], df['predicted_energy'], label='Predicted')

    # Future
    plt.plot(future_df['timestamp'], future_df['predicted_energy'], label='Future Forecast', linestyle='dashed')

    plt.legend()
    plt.title("Energy Forecasting (Past + Future)")
    plt.xlabel("Time")
    plt.ylabel("Energy")

    plt.savefig('images/future_forecast.png')
    plt.show()


def main():
    print("🚀 Running Forecast + Future Prediction...\n")

    df = load_data()
    model = load_model()

    df = predict_existing(df, model)

    last_timestamp = df['timestamp'].max()

    future_df = predict_future(model, last_timestamp)

    print("🔮 Future Predictions:\n")
    print(future_df.head())

    plot_all(df, future_df)

    # Save outputs
    future_df.to_csv('outputs/future_predictions.csv', index=False)

    print("\n✅ Future forecasting completed!")


if __name__ == "__main__":
    main()