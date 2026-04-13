# ==============================
# AI Energy Forecasting Project
# Phase 4: Model Training
# ==============================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib


def load_data(file_path):
    """
    Load processed dataset
    """
    df = pd.read_csv(file_path)
    print("✅ Processed data loaded!\n")
    return df


def prepare_features(df):
    """
    Prepare input features and target variable
    """
    X = df[['hour', 'day', 'month', 'day_of_week']]
    y = df['energy']

    print("✅ Features and target prepared!\n")
    return X, y


def split_data(X, y):
    """
    Split data into train and test sets
    (IMPORTANT: shuffle=False for time series)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print("✅ Data split into train and test!\n")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train Random Forest model
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("✅ Model trained successfully!\n")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("📊 Model Evaluation:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}\n")

    return y_pred


def save_model(model, path):
    """
    Save trained model
    """
    joblib.dump(model, path)
    print(f"💾 Model saved at: {path}\n")


def main():
    print("🚀 Starting Model Training...\n")

    # Step 1: Load data
    df = load_data('data/processed_energy.csv')

    print("📊 Data Preview:\n")
    print(df.head(), "\n")

    # Step 2: Prepare features
    X, y = prepare_features(df)

    # Step 3: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 4: Train model
    model = train_model(X_train, y_train)

    # Step 5: Evaluate model
    y_pred = evaluate_model(model, X_test, y_test)

    # Step 6: Save model
    save_model(model, 'models/energy_model.pkl')

    print("🎉 Model Training Completed!")


if __name__ == "__main__":
    main()