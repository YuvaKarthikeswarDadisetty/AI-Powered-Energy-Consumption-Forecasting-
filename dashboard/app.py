import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# Fix paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "processed_energy.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "energy_model.pkl")

st.set_page_config(page_title="Energy Forecasting", layout="wide")

st.title("⚡ AI Energy Consumption Forecasting Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()

# Sidebar
st.sidebar.header("Controls")
hours = st.sidebar.slider("Forecast Hours", 1, 48, 24)

# Predictions
X = df[['hour', 'day', 'month', 'day_of_week']]
df['predicted'] = model.predict(X)

# Future forecast
last_timestamp = df['timestamp'].max()

future_dates = pd.date_range(start=last_timestamp, periods=hours+1, freq='H')[1:]

future_df = pd.DataFrame({'timestamp': future_dates})
future_df['hour'] = future_df['timestamp'].dt.hour
future_df['day'] = future_df['timestamp'].dt.day
future_df['month'] = future_df['timestamp'].dt.month
future_df['day_of_week'] = future_df['timestamp'].dt.dayofweek

future_X = future_df[['hour', 'day', 'month', 'day_of_week']]
future_df['predicted'] = model.predict(future_X)

# Plot
fig = plt.figure(figsize=(12,5))

plt.plot(df['timestamp'], df['energy'], label='Actual')
plt.plot(df['timestamp'], df['predicted'], label='Predicted')
plt.plot(future_df['timestamp'], future_df['predicted'], linestyle='dashed', label='Future')

plt.legend()
plt.title("Energy Forecast")

st.pyplot(fig)

# Table
st.subheader("📊 Future Predictions")
st.dataframe(future_df.head(20))