import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/energy_full.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

plt.figure(figsize=(10,5))
plt.plot(df['timestamp'], df['energy'])

plt.title("Energy Consumption Trend")
plt.xlabel("Time")
plt.ylabel("Energy")

plt.savefig('images/data_trend.png')
plt.show()