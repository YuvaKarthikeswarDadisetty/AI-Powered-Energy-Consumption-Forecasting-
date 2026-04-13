import pandas as pd
import numpy as np

# Create 1000 hourly timestamps
date_range = pd.date_range(start='2023-01-01', periods=1000, freq='H')

# Create realistic energy pattern
np.random.seed(42)

energy = (
    200
    + 50 * np.sin(np.linspace(0, 50, 1000))   # daily pattern
    + np.random.normal(0, 10, 1000)           # noise
)

df = pd.DataFrame({
    'timestamp': date_range,
    'energy': energy
})

df.to_csv('data/energy_full.csv', index=False)

print("✅ energy_full.csv created successfully")