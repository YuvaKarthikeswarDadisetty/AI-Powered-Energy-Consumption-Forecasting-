import pandas as pd

df = pd.read_csv('data/energy.csv')

print("DATA PREVIEW:")
print(df.head())

print("\nDATA TYPES BEFORE:")
print(df.dtypes)

df['timestamp'] = pd.to_datetime(df['timestamp'])

print("\nDATA TYPES AFTER:")
print(df.dtypes)