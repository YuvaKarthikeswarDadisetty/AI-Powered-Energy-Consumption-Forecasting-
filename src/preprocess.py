# ==============================
# AI Energy Forecasting Project
# Phase 3: Data Preprocessing
# ==============================

import pandas as pd

def load_data(file_path):
    """
    Load dataset from CSV file
    """
    try:
        df = pd.read_csv(file_path)
        print("✅ Data loaded successfully!\n")
        return df
    except Exception as e:
        print("❌ Error loading data:", e)
        return None


def convert_datetime(df):
    """
    Convert timestamp column to datetime format
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print("✅ Timestamp converted to datetime\n")
    return df


def check_missing_values(df):
    """
    Check and handle missing values
    """
    print("🔍 Checking missing values...\n")
    print(df.isnull().sum())

    # Drop missing values if any
    df = df.dropna()

    print("\n✅ Missing values handled\n")
    return df


def sort_data(df):
    """
    Sort dataset based on timestamp
    """
    df = df.sort_values(by='timestamp')
    print("✅ Data sorted by timestamp\n")
    return df


def feature_engineering(df):
    """
    Extract useful time-based features
    """
    print("⚙️ Performing feature engineering...\n")

    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    print("✅ Feature engineering completed\n")
    return df


def save_processed_data(df, output_path):
    """
    Save processed dataset
    """
    df.to_csv(output_path, index=False)
    print(f"✅ Processed data saved at: {output_path}\n")


def main():
    print("🚀 Starting Data Preprocessing...\n")

    # Step 1: Load data
    df = load_data('data/energy_full.csv')

    if df is None:
        return

    print("📊 Initial Data Preview:\n")
    print(df.head(), "\n")

    # Step 2: Convert datetime
    df = convert_datetime(df)

    # Step 3: Handle missing values
    df = check_missing_values(df)

    # Step 4: Sort data
    df = sort_data(df)

    # Step 5: Feature Engineering
    df = feature_engineering(df)

    print("📊 Processed Data Preview:\n")
    print(df.head(), "\n")

    # Step 6: Save processed data
    save_processed_data(df, 'data/processed_energy.csv')

    print("🎉 Preprocessing Completed Successfully!")


if __name__ == "__main__":
    main()